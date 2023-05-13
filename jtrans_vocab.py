from bert_pytorch import WordVocab

if __name__ == '__main__':
    dataset_path = './data/paragraphs_pair_new.txt'
    output_path = './data/paragraphs_pair_new.pkl'
    specials = ['<unk_relate>']

    with open(dataset_path, "r", encoding='utf-8') as f:
        vocab = WordVocab(f, specials=specials)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(output_path)
