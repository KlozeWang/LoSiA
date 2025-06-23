import itertools

import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import *
import re


class PreprocessedIterableDataset_TrainOnInputs(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


IGNORE_INDEX=-100


class PreprocessedIterableDataset_TrainOnLabels(IterableDataset):
    def __init__(self,
                 raw_dataset,
                 tokenizer,
                 batch_size: int = 1,
                 max_length: Optional[int] = 512,
                 truncate_method: Optional[str] = "tail",
                 pad_max_len: Optional[bool] = False,
                ):
        # assert hasattr(raw_dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {raw_dataset}"
        # assert hasattr(raw_dataset, "__len__"), f"The dataset must have __len__ method. dataset is {raw_dataset}"
        self.raw_dataset = raw_dataset
        self.batch_size = batch_size

        self.tokenizer = tokenizer
        self.truncate_method = truncate_method
        self.max_seq_length = max_length
        self.not_pad_max_len = False if pad_max_len else True
        assert self.truncate_method == "tail", print("only tail truncate support")
    
    def tokenize_example(self, example):
        labels = []
        tokenized_ids = []
        tokenized_input = self.tokenizer(example["input"], add_special_tokens=False)
        tokenized_output = self.tokenizer(example["output"], add_special_tokens=False)
        tokenized_ids += tokenized_input["input_ids"]
        labels += [IGNORE_INDEX] * len(tokenized_input["input_ids"])
        tokenized_ids += tokenized_output["input_ids"]
        labels += tokenized_output["input_ids"]
        assert len(tokenized_ids) == len(labels)

        return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}, len(tokenized_input["input_ids"])

    def pad_truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        tokenized_example["attention_mask"] = torch.LongTensor([0]*len(tokenized_example["input_ids"]))
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]
            tokenized_example["len"] = self.max_seq_length
        elif old_len <= self.max_seq_length:
            tokenized_example["input_ids"] = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]*(self.max_seq_length - old_len)), tokenized_example["input_ids"]])
            tokenized_example["labels"] = torch.cat([torch.LongTensor([IGNORE_INDEX]*(self.max_seq_length - old_len)), tokenized_example["labels"]])
            tokenized_example["attention_mask"] = torch.LongTensor([0]*(self.max_seq_length - old_len) + [1]*old_len)
            tokenized_example["len"] = old_len
        assert len(tokenized_example["input_ids"]) == len(tokenized_example["labels"]) == len(tokenized_example["attention_mask"]) == self.max_seq_length
        return tokenized_example

    def _format_batch(self, batch):
        if self.not_pad_max_len:
            pad_len = max([item["len"] for item in batch])
            pad_len = min(max(((pad_len//8) + 1) * 8, 256), self.max_seq_length)
        else:
            pad_len = self.max_seq_length

        input_ids = torch.stack([item["input_ids"].squeeze(0)[self.max_seq_length - pad_len:] for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0)[self.max_seq_length - pad_len:] for item in batch])
        labels = torch.stack([item["labels"].squeeze(0)[self.max_seq_length - pad_len:] for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.raw_dataset)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.raw_dataset, worker_id, None, num_workers)
        
        batch = []
        for example in iter_data:
            tokenized_example, lenl = self.tokenize_example(example)

            if lenl >= self.max_seq_length:
                continue
            tokenized_example = self.pad_truncate(tokenized_example)
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
        
        if batch:
            yield self._format_batch(batch)

    def __len__(self):
        return (len(self.raw_dataset) + self.batch_size - 1) // self.batch_size
