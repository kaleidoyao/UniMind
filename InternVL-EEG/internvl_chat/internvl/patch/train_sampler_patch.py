from typing import List, Optional

import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import LengthGroupedSampler, RandomSampler, has_length
import torch.distributed as dist
import random


def group_and_randomize(A, size, B):
    chunks = []
    start_value = 0
    for end_value in B:
        chunk = [x for x in A if start_value <= x < end_value]
        chunks.append(chunk)
        start_value = end_value

    chunked_data = []
    for chunk in chunks:
        random.shuffle(chunk)
        chunk_subchunks = [chunk[i:i + size] for i in range(0, len(chunk), size) if len(chunk[i:i + size]) == size]
        chunked_data.append(chunk_subchunks)

    result = chunked_data[0]
    for i in range(1, len(chunked_data)):
        new_chunks = chunked_data[i]
        current_length = len(result)
        insert_positions = [int(j * current_length / len(new_chunks)) for j in range(len(new_chunks))]
        for pos, subchunk in zip(insert_positions, new_chunks):
            result.insert(pos, subchunk)

    flattened_result = []
    for sublist in result:
        flattened_result.extend(sublist)
    return flattened_result


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks == 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0] * num_chunks
    num_indices_per_chunk = len(indices) // num_chunks
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i:i+megabatch_size].tolist() for i in range(0, len(indices), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    all_indices = []
    for megabatch in megabatches:
        for chunk in megabatch:
            all_indices.extend(chunk)
    return all_indices


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
        dataset_range=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        self.dataset_range = dataset_range
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = get_rank()
        if lengths is None:
            model_input_name = model_input_name or "input_ids"
            if not (isinstance(dataset[0], dict) or model_input_name not in dataset[0]):
                raise ValueError("Can only infer lengths for datasets with 'input_ids' key.")
            lengths = [len(feat[model_input_name]) for feat in dataset]
        elif isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return (len(self.lengths) + self.world_size - 1) // self.world_size

    def __iter__(self):
        all_indices = get_length_grouped_indices(
            self.lengths, self.batch_size, self.world_size, self.generator
        )
        total_samples = len(all_indices)
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank
        if self.rank == self.world_size - 1:
            end_idx = total_samples
        rank_indices = all_indices[start_idx:end_idx]
        rank_indices = group_and_randomize(rank_indices, self.batch_size, self.dataset_range)
        return iter(rank_indices)


def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None
    if self.args.group_by_length:
        lengths = []
        for dataset in self.train_dataset.datasets:
            lengths = lengths + dataset.length
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
            dataset_range=getattr(self.args, "dataset_range", None),
        )
    else:
        return RandomSampler(self.train_dataset)


def replace_train_sampler():
    transformers.Trainer._get_train_sampler = _get_train_sampler
