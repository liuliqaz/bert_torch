from bert_pytorch import BERT, BERTLM, ScheduledOptim, WordVocab, JTransDataset, BERTTrainer, JTransTrainer
from load_data import load_data_wiki
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import os


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def get_gpus_by_ids(ids):
    devices = [torch.device(f'cuda:{i}')
             for i in ids]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
    vocab_path = './data/jtrans_x86.pkl'
    relate_vocab_path = './data/relate_new.pkl'
    train_data_path = '../jTrans_pair_tiny.txt'

    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Relation Vocab", relate_vocab_path)
    relate_vocab = WordVocab.load_vocab(relate_vocab_path)
    print("Relation Vocab Size: ", len(relate_vocab))

    print("Loading Train Dataset", train_data_path)
    train_dataset = JTransDataset(train_data_path, vocab, relate_vocab, seq_len=64, corpus_lines=None, on_memory=True)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)

    for i in range(len(train_dataset)):
        b = train_dataset[i]
        print(b)
        break
    # for i, data in enumerate(train_data_loader):
    #     b = {key: value for key, value in data.items()}
    #     print(b["bert_input"].shape)
    #     # print(b)
    #     break

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=768, n_layers=12, attn_heads=12)

    print("Creating BERT Trainer")
    # devices = try_all_gpus()
    devices = get_gpus_by_ids([0,4])
    print("Training device:", devices)
    with_cuda = True
    trainer = JTransTrainer(bert, len(vocab), len(relate_vocab), train_dataloader=train_data_loader,
                            test_dataloader=None, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
                            with_cuda=with_cuda, cuda_devices=devices, log_freq=10)

    print("Training Start")
    epochs = 10
    save_path = './saved_model'
    save_model = 'bert.model'
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.save(epoch, os.path.join(save_path, save_model))
