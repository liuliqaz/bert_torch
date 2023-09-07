from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import RobertaProcessing
from transformers.models.bert import BertTokenizer, WordpieceTokenizer
from tqdm import tqdm
import json
import argparse
from transformers import AutoTokenizer
import collections
from datasets import Dataset, DatasetDict


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[no_rela]"]
UNK_TOKENS = "[UNK]"
BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def prepare_tokenizer_trainer():
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKENS))
    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS)

    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer, trainer


def train_tokenizer(files, json_path, vocab_path):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer()
    tokenizer.train(files, trainer)  # training the tokenzier
    tokenizer.post_processor = RobertaProcessing(
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
        cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
    )

    tokenizer.save(json_path)
    with open(json_path, 'r') as f:
        vocab_json = json.load(f)
    vocab = vocab_json['model']['vocab']
    vocab = list(vocab.keys())
    with open(vocab_path, 'w') as f:
        for item in tqdm(vocab):
            f.write("%s\n" % item)


def main():
    parser = argparse.ArgumentParser(description="data tokenize trainer.")
    parser.add_argument("--input_path", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/pretrain.txt')
    parser.add_argument("--json_path", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/tmp_json.json')
    parser.add_argument("--vocab_path", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/tmp_vocab.txt')
    parser.add_argument("--output_dir", type=str, default='/home/liu/bcsd/datasets/edge_gnn_datas/tokenizer')
    args = parser.parse_args()

    input_path = args.input_path
    json_path = args.json_path
    vocab_path = args.vocab_path
    output_dir = args.output_dir

    # test
    # input_path = './data/paragraphs_pair_new.txt'
    # json_path = './data/tmp_json.json'
    # vocab_path = './data/tmp_vocab.txt'
    # output_dir = './data/tokenizer'

    files = [input_path]
    train_tokenizer(files, json_path, vocab_path)
    tokenizer = BertTokenizer(vocab_path)
    tokenizer.save_pretrained(output_dir)


def test():
    pretrained_tok_path = '/home/liu/bcsd/datasets/test_data/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tok_path, do_lower_case=False, do_basic_tokenize=False, use_fast=False)
    tokenizer.do_basic_tokenize = False
    # tokenizer = Tokenizer.from_file('/home/liu/bcsd/datasets/test_data/tmp_json.json')
    sen = 'cbz w4 const_hex [SEP] mov w21 const_hex b jump_addr $va b.mi'

    tokens = tokenizer.tokenize(sen)

    ids = tokenizer.convert_tokens_to_ids(tokens)

    idd = tokenizer(sen)

    tok = tokenizer.convert_tokens_to_ids('const_hex')

    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    t2 = WordpieceTokenizer(load_vocab('/home/liu/bcsd/datasets/test_data/tokenizer/vocab.txt'), unk_token=UNK_TOKENS)
    tokens2 = t2.tokenize(sen)

    print('debug')


def get_rela_token():
    # pretrained_tok_path = '/home/liu/bcsd/datasets/edge_gnn_datas/tokenizer'
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_tok_path, do_lower_case=False, do_basic_tokenize=False, use_fast=False)
    # res_rela_token = []
    # for key in tokenizer.vocab.keys():
    #     if key[0] == '[' and key[-1] == ']' and key not in BERT_SPECIAL_TOKENS:
    #         res_rela_token.append(key + '\n')

    # vocab_path = '/home/liu/bcsd/bert_torch/rela_vocab.txt'
    # output_dir = '/home/liu/bcsd/datasets/edge_gnn_datas/rela_tokenizer'
    # tokenizer = BertTokenizer(vocab_path)
    # tokenizer.save_pretrained(output_dir)

    s = ['[f_j]', '[t_j]']
    p1 = '/home/liu/bcsd/datasets/edge_gnn_datas/tokenizer'
    p2 = '/home/liu/bcsd/datasets/edge_gnn_datas/rela_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(p2, use_fast=False, do_lower_case=False, do_basic_tokenize=False)
    ids = tokenizer.convert_tokens_to_ids(s)

    print('done')


def try_tokenize():
    p1 = '/home/liu/bcsd/datasets/edge_gnn_datas/tokenizer'
    p2 = '/home/liu/bcsd/datasets/edge_gnn_datas/rela_tokenizer'

    tokenizer = AutoTokenizer.from_pretrained(p1, use_fast=False, do_lower_case=False, do_basic_tokenize=False)
    rela_tokenizer = AutoTokenizer.from_pretrained(p2, use_fast=False, do_lower_case=False, do_basic_tokenize=False)

    def tokenize_function(examples):
        result = tokenizer(
            # examples['sentence'],
            examples['0'],
            examples['2'],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )
        result['rela_token'] = rela_tokenizer.convert_tokens_to_ids(examples['rela'])
        # result['rela_token_idx'] = [i + 1 for i in examples['sep']] # use [SEP] as predict token
        return result


    read_file = '/home/liu/bcsd/datasets/test_data/pretrain_with_rand_pair_sep.txt'
    with open(read_file, 'r') as f:
        json_str = f.read()
    parse_json = json.loads(json_str)
    data_list = parse_json['train'][:10]
    raw_datasets = DatasetDict({
        'train': Dataset.from_list(data_list),
    })
    raw_datasets = raw_datasets.filter(lambda x: (len(x['0']) + len(x['2']) <= 512-3))

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        num_proc=1,
        desc="Running tokenizer on dataset line_by_line",
    )

    for index in range(len(tokenized_datasets)):
        ori = raw_datasets['train'][index]
        t = tokenized_datasets['train'][index]
        print(f"Sample {index} of the training set: {t}.")
        print(f"Ori Sample {index} of the training set: {ori}.")


if __name__ == '__main__':
    # main()

    # test()

    # get_rela_token()

    try_tokenize()