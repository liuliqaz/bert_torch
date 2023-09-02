import json
import math
import datasets
import torch
import random
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


if __name__ == '__main__':
    file = '/home/liu/bcsd/datasets/test_data/pretrain_with_rand_pair.txt'
    with open(file, 'r') as f:
        json_str = f.read()
    parse_json = json.loads(json_str)
    data_list = parse_json['train'][:1000]
    train_dataset = Dataset.from_list(data_list[math.ceil(len(data_list)*0.01):])
    eval_dataset = Dataset.from_list(data_list[:math.ceil(len(data_list)*0.01)])
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset,
    })

    pretrained_tok_path = '/home/liu/bcsd/datasets/test_data/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tok_path, use_fast=False, do_lower_case=False, do_basic_tokenize=False)

    def tokenize_function(examples):
        result = tokenizer(
            examples['sentence'],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_special_tokens_mask=True,
        )
        result['rela_token'] = tokenizer.convert_tokens_to_ids(examples['rela'])
        result['rela_token_idx'] = [i + 1 for i in examples['sep']]
        # result['special_tokens_mask'][examples['sep'] + 1] = 1
        # if 'x86' in examples['arch']:
        #     result['arch'] = [1 for _ in range(100)]
        # if 'arm' in examples['arch']:
        #     result['arch'] = [2 for _ in range(100)]
        # if 'mips' in examples['arch']:
        #     result['arch'] = [3 for _ in range(100)]
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
    )

    tokenized_eval_datasets = eval_dataset.map(
        tokenize_function,
        batched=True
    )


    for index in range(3):
        token_sample = tokenized_datasets['train'][index]
        eval_token_sample = tokenized_eval_datasets[index]
        pass

    print('pass')

    
