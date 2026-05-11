import argparse
import ast
import difflib
import itertools
import json
import os
import pickle
import random
import time
from collections import OrderedDict
from functools import partial
from typing import Optional

import numpy as np
import safetensors.torch
import torch
from internvl.conversation import get_conv_template
from internvl.model import load_model_and_tokenizer
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDatasetEEG, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from scipy.signal import resample
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             f1_score, recall_score)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
IGNORE_INDEX = -100

def evaluate_answer(answer, annotations, label):
    correct = 0
    sim_answer_annotation = difflib.SequenceMatcher(None, answer[0], annotations[0]).ratio()
    similarities = [difflib.SequenceMatcher(None, answer[0], lbl).ratio() for lbl in label]
    max_similarity = max(similarities)
    max_similarity_label = label[similarities.index(max_similarity)]

    if sim_answer_annotation == max_similarity:
        correct = 1
    else:
        correct = 0

    return correct, max_similarity_label


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    ch_names = [_['ch_names'] for _ in batches]
    task_caption = [_['task_caption'] for _ in batches]

    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(questions, return_tensors='pt', padding=True)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    return pixel_values, questions, input_ids, attention_mask, question_ids, annotations, ch_names, task_caption


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, train, test, prompt, few_shot, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, ch_names=None, sample_rate=200, task_caption=None, ds_name=None, mean_value=None, std_value=None,
                 max_value=None, min_value=None, normalize_type=None, num_image_token=None, resample_method=None, num_query_tokens=None, num_cquery_tokens=None, num_tquery_tokens=None, tokenizer_path=None):
        self.tokenizer = tokenizer
        self.test = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        if few_shot > 0:
            self.train = open(train).readlines()
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.ch_names = ch_names
        self.sample_rate = sample_rate
        self.default_rate = 200
        self.ds_name = ds_name
        self.mean_value = mean_value
        self.std_value = std_value
        self.max_value = max_value
        self.min_value = min_value
        self.normalize_type = normalize_type
        self.task_caption = task_caption
        self.template_name = "internlm2-chat"
        self.num_query_tokens = num_query_tokens
        self.num_cquery_tokens = num_cquery_tokens
        self.num_tquery_tokens = num_tquery_tokens
        self.group_by_length = True

        self.conv_template = get_conv_template(self.template_name)
        self.system_message = self.conv_template.system_message
        self.base_prompt = 'Answer the question using a single word or phrase.'
        self.vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
        # infovqa_prompt = 'Answer the question directly.'
        self.infovqa_prompt = 'Answer the question using a single word or phrase.'
        self.ai2d_prompt = ''
        if resample_method in ["qformer", "c2tqformer"]:
            self.num_image_token = num_query_tokens
        elif 'c2tqformer_v2' in resample_method:
            self.num_image_token = num_query_tokens * 2 + 1
        elif resample_method in ["ctqformer"]:
            num_c = len(ch_names)
            num_T = int((num_image_token - 1) / num_c)
            self.num_image_token = (num_c + num_T) * num_query_tokens + 1
        elif "ctqformer_v2" in resample_method:
            num_c = len(ch_names)
            num_T = int((num_image_token - 1) / num_c)
            self.num_image_token = (num_c * num_tquery_tokens) + (num_T * num_cquery_tokens) + 1
            # print("aaaaaaaaaaaaaaaa")
            # print(self.num_image_token)
        elif resample_method == "c2tqformer_onlychannel":
            num_c = len(ch_names)
            num_T = int((num_image_token - 1) / num_c)
            self.num_image_token = (num_T) * num_cquery_tokens + 1
        elif resample_method == "channel_router":
            num_c = len(ch_names)
            num_T = int((num_image_token - 1) / num_c)
            self.num_image_token = num_T + 1
        else:
            if ds_name == "BCI2A":
                self.num_image_token = 1
            else:
                self.num_image_token =num_image_token
        pass

    def __len__(self):
        return len(self.test)
    
    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_eeg(self, image_path):
        sample = pickle.load(open(image_path, 'rb'))
        X = sample['X'] if 'X' in sample else sample
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if self.default_rate != self.sample_rate:
            b, sequence = X.shape
            T = int(sequence / self.sample_rate)
            new_length = T * self.default_rate
            X = resample(X, new_length, axis=1)
        return X

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image = data["EEG"]
        if '<image>' not in data['conversations'][0]['value']:
            data['conversations'][0]['value'] = '<image>\n' + data['conversations'][0]['value']
        question = data["conversations"][0]["value"]  # human的value
        question_id = data["id"]
        annotation = data["conversations"][1]["value"]  # gpt的value
        
        # image, question, question_id, annotation = data['image'], data[
        #     'question'], data['question_id'], data.get('answer', None)
       
        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        # image = Image.open(image).convert('RGB')
        image = self.load_eeg(image)
        if self.normalize_type == "zscore":
            if self.ds_name in ["TUSL", "TUAB", "TUEV", "Workload", "BCI2A", "SEEDIV", "SHHS", "SleepEDF"]: 
                mean_value, std_value = np.array(self.mean_value), np.array(self.std_value)
                mu, sigma = np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1)
                image = (image - mu) / (sigma + 1e-8)
                image = torch.from_numpy(image) 
            elif self.ds_name in ["SHU"]: 
                image = image / (np.quantile(np.abs(image), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
                image = torch.from_numpy(image)
            else:
                image = torch.from_numpy(image) 
        else:
            if self.ds_name == "TUAB":
                image = torch.from_numpy(image) 
                image = torch.nn.functional.normalize(image, p=2, dim=-1)
            elif self.ds_name in ["TUSL", "Workload", "TUEV"]:
                image = torch.from_numpy(image).float()
                mean = image.mean(dim=0)
                std = image.std(dim=0)
                image = (image - mean) / std
            else:
                image = torch.from_numpy(image)
        pixel_values = [image]
        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = [torch.from_numpy(image) for image in images]
        pixel_values = torch.stack(pixel_values)
     
        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()
        num_patches = pixel_values.size(0)
        # Preprocess the conversations and generate the return dictionary
        template = get_conv_template(self.template_name)
        template.system_message = self.system_message
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        question = data['conversations'][0]['value'] + self.base_prompt

        history = []
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

        task_tokenizer = self.tokenizer(
            self.task_caption,
            return_tensors='pt',
            padding='max_length',
            max_length=10,
            truncation=True,
        )
        return {
            'question_id': question_id,
            # 'input_ids': input_ids[0],
            # 'attention_mask': attention_mask[0],
            'question': query,
            'pixel_values': pixel_values,
            'annotation': annotation,
            'ch_names': self.ch_names,
            "task_caption": task_tokenizer,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response

def calculate_metrics_SEED(data):
    unique_labels = sorted(set(item['truth'] for item in data) | set(item['answer'] for item in data))
    # print(unique_labels)
    category_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_true = [item['truth'] for item in data]
    y_pred = [item['answer'] for item in data]
    y_true_encoded = [category_map[label] for label in y_true]
    y_pred_encoded = [category_map[label] for label in y_pred]
    # print("00000000000000000000")
    # print(y_true_encoded)
    # print(y_pred_encoded)

    classes = len(category_map)
    # balanced_accuracy = balanced_accuracy_score(y_true_encoded, y_pred_encoded)

    recalls = recall_score(y_true_encoded, y_pred_encoded, average=None, labels=np.arange(classes))
    balanced_accuracy = np.mean(recalls)
    cohen_kappa = cohen_kappa_score(y_true_encoded, y_pred_encoded)
    weighted_f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', labels=np.arange(classes))
    
    return {
        'BalAccuracy': balanced_accuracy,
        'CohenKappa': cohen_kappa,
        'WeightedF1': weighted_f1
    }


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    with open(args.test_dataset, 'r') as json_file:
        ds_collections = json.load(json_file)

    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            tokenizer = tokenizer,
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            ch_names = ds_collections[ds_name]['ch_names'],
            sample_rate = ds_collections[ds_name]['sample_rate'],
            task_caption = ds_collections[ds_name]['task_caption'],
            ds_name = ds_name,
            mean_value = ds_collections[ds_name].get('mean', None),
            std_value = ds_collections[ds_name].get('std', None),
            max_value = ds_collections[ds_name].get('max', None),
            min_value = ds_collections[ds_name].get('min', None),
            num_image_token = ds_collections[ds_name]['num_image_token'],
            normalize_type = args.normalize_type,
            resample_method = args.resample_method,
            num_query_tokens = args.num_query_tokens,
            num_cquery_tokens = args.num_cquery_tokens,
            num_tquery_tokens = args.num_tquery_tokens,
            tokenizer_path = args.checkpoint
        )
        # print("00000000000000000")
        # print(args.batch_size)
        # exit()
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        total=0
        correct=0
        for _, (pixel_values, questions, input_ids, attention_mask, question_ids, annotations, ch_names, task_caption) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            # print("000000000000000")
            # print(pixel_values.shape)
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # exit()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            preds, channel_static_list, time_static_list, attn_weights_c, attn_weights_t = model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                question=questions[0],
                generation_config=generation_config,
                verbose=True,
                task_caption = task_caption,
                resample_method = args.resample_method,
                num_query_tokens = args.num_query_tokens,
                use_fuse = args.use_fuse,
                ch_names = ch_names,
                ds_name = ds_name
            )

            total += len(preds)
            labels = ast.literal_eval(args.answer_labels)
            for pred, annotation, question, question_id in zip(preds, annotations, questions, question_ids):
                answers = [pred]
                correct_pred, anno = evaluate_answer(answers, [annotation], labels)
                answers = [anno]
                if correct_pred:
                    correct+=1
                print('correct:', correct, ' / ', 'total: ',total)
                print('original results:', pred)
                print('predicted results:',answers[0])
                print('ground truth:', annotation)
                i = 0
                # for question, question_id, answer, anno in zip(questions, question_ids, answers, annotation):
                if ds_name in ['SEED', 'HMC', 'Workload', 'TUAB', 'TUEV', 'TUSL', 'BCI2A', 'SEEDIV', 'SHU', "SHHS", "SleepEDF"]:
                    if 'ctqformer' not in args.resample_method:
                        outputs.append({
                            'question_id': question_id,
                            'answer': answers[0],
                            'truth': annotation
                        })
                    else:
                        outputs.append({
                            'question_id': question_id,
                            'answer': answers[0],
                            'truth': annotation,
                            'attn_weights_c': attn_weights_c[i],
                            'attn_weights_t': attn_weights_t[i],
                            "channel_query_indices": channel_static_list[i],
                            "time_query_indices": time_static_list[i],
                        })
                    i += 1
                    print(len(outputs))
                    metrics = calculate_metrics_SEED(outputs)
                    print(metrics)
                else:
                    raise NotImplementedError

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            checkpoint_path = args.checkpoint
            parent_folder = os.path.basename(os.path.dirname(checkpoint_path))
            checkpoint_name = os.path.basename(checkpoint_path)
            target_folder = os.path.join(args.out_dir, parent_folder)
            os.makedirs(target_folder, exist_ok=True)
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{checkpoint_name}_{ds_name}_{time_prefix}.json'
            results_file = os.path.join(target_folder, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))
            if ds_name in ['SEED', 'HMC', 'TUEV', 'TUSL', 'TUAB', 'Workload', 'BCI2A', 'SEEDIV', 'SHU']:
                metrics = calculate_metrics_SEED(outputs)
                print(metrics)
                summaries.append([args.checkpoint, ds_name, metrics['BalAccuracy'], metrics['CohenKappa'], metrics['WeightedF1']])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--normalize_type', type=str, default='old')
    parser.add_argument('--test_dataset', type=str, default='shell/data/internvl_1_2_test.json')
    parser.add_argument('--datasets', type=str,
                        default='okvqa_val,textvqa_val,vizwiz_val,ai2diagram_test,gqa_testdev_llava')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--use_fuse', type=bool, default=False)
    parser.add_argument('--resample_method', type=str, default='')
    parser.add_argument('--num_query_tokens', type=int, default=16)
    parser.add_argument('--num_cquery_tokens', type=int, default=4)
    parser.add_argument('--num_tquery_tokens', type=int, default=4)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--answer_labels', type=str, default='')
    parser.add_argument('--ds_name', type=str, default='')
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    args.ds_name = args.datasets
    # assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    model, tokenizer = load_model_and_tokenizer(args)

    # Load any extra parameters (e.g. layer-norm gamma) stored only in shard 1.
    extra_shard_path = os.path.join(args.checkpoint, 'model-00001-of-00004.safetensors')
    if os.path.exists(extra_shard_path):
        extra_shard = safetensors.torch.load_file(extra_shard_path)
        gamma_keys = OrderedDict(
            {k: v for k, v in extra_shard.items() if 'gamma' in k}
        )
        if gamma_keys:
            model.load_state_dict(gamma_keys, strict=False)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
