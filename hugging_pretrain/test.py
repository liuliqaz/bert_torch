import json
import math
import datasets
import torch
import random
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, CONFIG_MAPPING
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import *

def test_datasets():
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

def test_bert_encode():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    

    prompt = "In Italy, pizza served in formal settings."
    next_sentence = "The sky is blue due to the shorter wavelength of blue light."
    encoding = tokenizer(prompt, next_sentence, return_tensors="pt", padding="max_length", max_length=100)

    encoding2 = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=100)

    config = CONFIG_MAPPING['bert']()
    model = MyBertForNextSentencePrediction(config)
    outputs = model(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits
    assert logits[0, 0] < logits[0, 1]  # next sentence was random


class MyBertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == '__main__':
    test_bert_encode()
    
