#coding:utf-8
import numpy as np
from torchtext import data
from config import opt

'''
get the data from the tsv file.
'''

def get_data():
    pass

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def load_train():
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False, use_vocab=False)
    train_datafields = [("phrase_id", id_field), ("sen_id", None), ("text", text_field), ("label", label_field)]
    train = data.TabularDataset(path='data/train.tsv',
                                                  format='tsv',
                                                  skip_header=True,
                                                  fields=train_datafields)

    test_datafields = [("phrase_id", id_field), ("sen_id", None), ("text", text_field)]
    test = data.TabularDataset(path='data/test.tsv',
                               format='tsv',
                               skip_header=True,
                               fields=test_datafields)

    id_field.build_vocab(train)
    text_field.build_vocab(train, test)
    label_field.build_vocab(train)
    all_vocab = text_field.vocab

    train_iter = data.BucketIterator(
        dataset=train, batch_size=opt.batch_size,
        sort_key=lambda x: len(x.text),shuffle=True,repeat=False)

    return train_iter, all_vocab


def load_test():
    id_field = data.Field(sequential=False, use_vocab=False)
    text_field = data.Field(lower=True)
    tv_datafields = [("phrase_id", id_field), ("sen_id", None), ("text", text_field)]
    test = data.TabularDataset(path='data/test.tsv',
                                format='tsv',
                                skip_header=True,
                                fields=tv_datafields)

    text_field.build_vocab(test)
    id_field.build_vocab(test)

    test_iter = data.Iterator(
        dataset=test, batch_size=opt.batch_size,
        sort=False, train=False, sort_within_batch=False, repeat=False)

    return test_iter


def get_word_vec(train_vocab):
    word_to_idx = train_vocab.stoi
    pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(train_vocab), 300))
    pretrained_embeddings[0] = 0
    word2vec = load_bin_vec('data/GoogleNews-vectors-negative300.bin', word_to_idx)
    for word, vector in word2vec.items():
        pretrained_embeddings[word_to_idx[word] - 1] = vector #这里将<unk>去掉了，而<pad>的vec全是0
    return pretrained_embeddings


# #
if __name__ == "__main__":
    trn, train_vocab = load_train()
    print(trn.__len__())
    import tqdm
    count = 0
    for batch in trn:
        count+=1
        label = batch.label
#         # label.data.sub_(1)
#         print(label)
# #      # batch.label.data.sub_(1)
        print(batch.text.data.shape)
        if count == 1561:
            print(batch.text.data)

# # pe = get_word_vec(train_vocab)
# # print(pe[0],pe[2])
# # print(pe.shape)
