# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import math
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, InternVisionModel_EEG
from .utils import log_optimal_transport

logger = logging.get_logger(__name__)

# Routing statistics accumulators — populated during evaluation to analyse query pool usage.
time_static_list = [0] * 32
channel_static_list = [0] * 32
def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.contrastive_method = getattr(config, 'contrastive_method', 'none')
        self.resample_method = getattr(config, 'resample_method', 'none')
        self.num_query_tokens = getattr(config, 'num_query_tokens', 16)
        self.num_cquery_tokens = getattr(config, 'num_cquery_tokens', 1)
        self.num_tquery_tokens = getattr(config, 'num_tquery_tokens', 1)
        self.num_query_pool = getattr(config, 'num_query_pool', 32)
        self.dataset_info = getattr(config, 'dataset_info', 'none')
        self.task_emb_len = getattr(config, 'task_emb_len', 10)

        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        self.num_experts = 3
        # logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model=InternVisionModel_EEG(config.vision_config)
            # self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        # LaBraM encoder output dim is 1152; kept constant rather than reading from config.
        vit_hidden_size = 1152
        llm_hidden_size = config.llm_config.hidden_size

        if self.resample_method == "ctqformer_v2_2router_merge":
            num_query_pool = self.num_query_pool
            self.num_cquery_tokens = math.ceil(self.num_cquery_tokens / 2)
            self.num_tquery_tokens = math.ceil(self.num_tquery_tokens / 2)
            self.ctqformer_query_c_single = nn.ParameterDict()
            self.ctqformer_query_t_single = nn.ParameterDict()
            for dataset_name in self.dataset_info:
                self.ctqformer_query_c_single[dataset_name] = nn.Parameter(
                    torch.randn(1, 1, vit_hidden_size)
                )
                self.ctqformer_query_t_single[dataset_name] = nn.Parameter(
                    torch.randn(1, 1, vit_hidden_size)
                )
            self.query_pool_c = nn.Parameter(torch.randn(1, num_query_pool, vit_hidden_size))
            self.query_pool_t = nn.Parameter(torch.randn(1, num_query_pool, vit_hidden_size))
            self.time_router_manager = QueryRouterManager(self.dataset_info, vit_hidden_size, num_query_pool, single=True)
            self.channel_router_manager = QueryRouterManager(self.dataset_info, vit_hidden_size, num_query_pool, single=True)
            # self.ctqformer_query_c = nn.Parameter(torch.randn(1, num_cquery_tokens, vit_hidden_size)) 
            # self.ctqformer_query_t = nn.Parameter(torch.randn(1, num_tquery_tokens, vit_hidden_size))
            self.ctqformer_attn_c = nn.MultiheadAttention(embed_dim=vit_hidden_size, num_heads=8, batch_first=True)
            self.ctqformer_attn_t = nn.MultiheadAttention(embed_dim=vit_hidden_size, num_heads=8, batch_first=True)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def min_max_normalization(self, matrix):
        min_val = matrix.min()
        max_val = matrix.max()
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,#torch.Size([1, 872])
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            ch_id=None,
            task_caption=None,
            text_answer=None,
            answer_mask = None,
            ds_name = None,
            eeg_id=None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        ds_name = ds_name[0][0]
        vit_batch_size = pixel_values.shape[0]
        taskcaption_embeds = self.language_model.get_input_embeddings()(task_caption[0]['input_ids']).clone()
        answer_embeds = self.language_model.get_input_embeddings()(text_answer).clone()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        ch_names = ch_id[0]
        vit_embeds = self.extract_feature_eeg(pixel_values, ch_names, ds_name, taskcaption_embeds)
        vit_embeds = vit_embeds[image_flags == 1]  # torch.Size([b, seq, 4096]) torch.Size([1, 1, 256, 4096]) torch.Size([2, 256, 4096])

        B, N, C = input_embeds.shape  # 1 872 4096  torch.Size([1, 360, 4096])
        input_embeds = input_embeds.reshape(B * N, C)  # torch.Size([1, 360, 4096])
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
        # dynamic ViT batch size: 1, images per sample: 1.0, dynamic token length: 360
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
    
    
    def extract_feature_eeg(self, eeg_values, ch_names, ds_name, taskcaption_embeds=None, test=False):
        eeg_embeds = self.vision_model(eeg_values, ch_names)
                
        if self.resample_method == "ctqformer_v2_2router_merge":
            bs, seq, emb = eeg_embeds.shape
            eeg_embeds = F.normalize(eeg_embeds, dim=-1)
            c = len(ch_names)
            T = int((seq - 1)/c)
            time_query_route = self.time_router_manager.to(eeg_embeds.device)(eeg_embeds)
            channel_query_route = self.channel_router_manager.to(eeg_embeds.device)(eeg_embeds)
            
            channel_query_score = channel_query_route.cpu().tolist()
            time_query_score = time_query_route.cpu().tolist()
            top_t_indices = torch.topk(time_query_route, self.num_tquery_tokens, dim=-1).indices
            top_c_indices = torch.topk(channel_query_route, self.num_cquery_tokens, dim=-1).indices
            for item in top_c_indices:
                for x in item:
                    if x < len(channel_static_list ):
                        channel_static_list[x] += 1
            for item in top_t_indices:
                for x in item:
                    if x < len(time_static_list ): 
                        time_static_list[x] += 1
            ctqformer_query_c = self.query_pool_c[:, top_c_indices, :].squeeze(0)
            ctqformer_query_t = self.query_pool_t[:, top_t_indices, :].squeeze(0)
            query_c_single = self.ctqformer_query_c_single[ds_name].expand(bs * T, -1, -1) 
            query_t_single = self.ctqformer_query_t_single[ds_name].expand(bs * c, -1, -1) 
            cls_token, eeg_features = eeg_embeds[:, :1, :], eeg_embeds[:, 1:, :] 

            eeg_features = eeg_features.view(bs, c, T, emb) 
            eeg_features_c = eeg_features.permute(0, 2, 1, 3).reshape(bs * T, c, emb) 
            query_c = ctqformer_query_c.repeat(T, 1, 1) 
            query_c = torch.cat([query_c_single, query_c], dim=1)
            attn_c,  attn_weights_c = self.ctqformer_attn_c(query_c, eeg_features_c, eeg_features_c) 
            attn_weights_c = attn_weights_c.cpu().tolist()
            attn_c = attn_c.reshape(bs, -1, emb)
            eeg_features_t = eeg_features.reshape(bs * c, T, emb) 
            query_t = ctqformer_query_t.repeat(c, 1, 1) 
            query_t = torch.cat([query_t_single, query_t], dim=1)
            attn_t, attn_weights_t = self.ctqformer_attn_t(query_t, eeg_features_t, eeg_features_t)
            attn_weights_t = attn_weights_t.cpu().tolist()

            attn_t = attn_t.reshape(bs, -1, emb)
            eeg_out = torch.cat([cls_token, attn_c, attn_t], dim=1)  # [bs, C+T+1, emb]
            eeg_out = self.mlp1(eeg_out)
            if test:
                return eeg_out, channel_query_score, time_query_score, attn_weights_c, attn_weights_t
            else:
                return eeg_out
        else:
            eeg_embeds = self.mlp1(eeg_embeds.unsqueeze(1))
            return eeg_embeds, None, None, None

    def batch_chat(self, tokenizer, pixel_values, input_ids, attention_mask, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, task_caption=None, ch_names=None, use_fuse=False, resample_method=None, num_query_tokens=32, ds_name='none'):
        self.resample_method = resample_method
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        attention_mask = attention_mask.cuda()
        generation_config['eos_token_id'] = eos_token_id

        generation_output, channel_static_list, time_static_list, attn_weights_c, attn_weights_t = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_caption=task_caption,
            ch_names=ch_names[0],
            ds_name = ds_name,
            **generation_config
        )
        # response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

        responses = [response.split(template.sep)[0].strip() for response in responses]
        if return_history:
            history.append((question, response))
            return responses, history
        else:
            return responses, channel_static_list, time_static_list, attn_weights_c, attn_weights_t

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, task_caption=None, ch_names=None, use_fuse=False, resample_method=None, num_query_tokens=32, ds_name='none'):
        self.resample_method = resample_method
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        self.num_image_token = int(pixel_values.shape[-1] / 200 * len(ch_names[0])) + 1

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()

        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_caption=task_caption,
            ch_names=ch_names[0],
            ds_name = ds_name,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            task_caption=None,
            return_dict: Optional[bool] = None,
            ch_names=None,
            ds_name = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        device = self.language_model.device 
        input_ids = input_ids.to(device) 
        taskcaption_embeds = self.language_model.get_input_embeddings()(task_caption[0]['input_ids'].to(pixel_values.device)).clone()
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, channel_static_list, time_static_list, attn_weights_c, attn_weights_t = self.extract_feature_eeg(pixel_values, ch_names, ds_name, taskcaption_embeds, test=True)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)#92546
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                    f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs, channel_static_list, time_static_list, attn_weights_c, attn_weights_t

class RouterManager(nn.Module):
    """Per-dataset channel router that selects a subset of EEG channels."""

    def __init__(self, dataset_info, vit_hidden_size):
        super().__init__()
        self.routers = nn.ModuleDict()
        for dataset_name, dataset in dataset_info.items():
            num_experts = len(dataset["ch_names"])
            self.routers[dataset_name] = Mlp(vit_hidden_size, vit_hidden_size * 4, num_experts)

    def forward(self, x, dataset_name):
        return self.routers[dataset_name](x)


class QueryRouterManager(nn.Module):
    """Routes EEG embeddings to a learned query pool (shared or per-dataset)."""

    def __init__(self, dataset_info, vit_hidden_size, num_query_pool, single=False):
        super().__init__()
        self.routers = nn.ModuleDict()
        self.single = single
        if self.single:
            self.router = Mlp(vit_hidden_size, vit_hidden_size * 4, num_query_pool)
        else:
            for dataset_name, dataset in dataset_info.items():
                self.routers[dataset_name] = Mlp(vit_hidden_size, vit_hidden_size * 4, num_query_pool)

    def forward(self, x, dataset_name=None):
        if self.single:
            return self.router(x)
        else:
            return self.routers[dataset_name](x)


class TaskQueryRouterManager(nn.Module):
    """Task-conditioned query router (single shared router or per-dataset)."""

    def __init__(self, dataset_info, vit_hidden_size, num_query_pool, single=False):
        super().__init__()
        self.routers = nn.ModuleDict()
        self.single = single
        if self.single:
            self.routers["single"] = Mlp(vit_hidden_size, vit_hidden_size, num_query_pool)
        else:
            for dataset_name, dataset in dataset_info.items():
                self.routers[dataset_name] = Mlp(vit_hidden_size, vit_hidden_size * 4, num_query_pool)

    def forward(self, x, dataset_name=None):
        if self.single:
            return self.routers["single"](x)
        else:
            return self.routers[dataset_name](x)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.mean(x, dim=1) 
        return x
