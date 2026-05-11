import math
import os

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel


def rename_state_dict_keys(state_dict):
    """Fix nested LoRA key names so they match the model's parameter tree."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "base_model.model.base_model.model" in key:
            new_key = key.replace("base_model.model.base_model.model", "base_model.model")
            new_state_dict[new_key] = value
    return new_state_dict


def split_model(num_layers, vit_alpha=0.5):
    """Build a device_map that distributes transformer layers across all GPUs.

    GPU 0 hosts the vision model and a proportionally smaller share of LLM layers
    because it also handles the ViT embedding computation.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers_per_gpu = math.ceil(num_layers / (world_size - vit_alpha))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - vit_alpha))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map


def load_model_and_tokenizer(args):
    """Load InternVLChatModel and tokenizer from a safetensors checkpoint directory."""
    if args.auto:
        config = InternVLChatConfig.from_pretrained(args.checkpoint)
        num_hidden_layers = config.llm_config.num_hidden_layers
        device_map = split_model(num_hidden_layers)
    kwargs = {'device_map': device_map} if args.auto else {}

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)

    safetensor_files = sorted([f for f in os.listdir(args.checkpoint) if f.endswith(".safetensors")])
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {args.checkpoint}")

    state_dict = {}
    for file in safetensor_files:
        file_path = os.path.join(args.checkpoint, file)
        print(f"Loading {file_path} ...")
        raw_state_dict = load_file(file_path)
        state_dict.update(rename_state_dict_keys(raw_state_dict))

    model = InternVLChatModel.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()

    model.load_state_dict(state_dict, strict=False)
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        model = model.cuda()

    return model, tokenizer
