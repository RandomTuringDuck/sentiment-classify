#coding:utf8
import sys,os
import torch as t
from torch import nn
from torch.autograd import Variable
import tqdm
from torchnet import meter
import ipdb
from models.LSTM import LSTMSentiment
from dataloader import load_train,load_test,get_word_vec
from config import opt




# 将结果写入csv文件
def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['PhraseId','Sentiment'])
        writer.writerows(results)


# train the model LSTM
def train(**kwargs):
    # fire提供的参数
    for k, v in kwargs.items():
        setattr(opt, k, v)

    trn, train_vocab = load_train()
    word_vec = get_word_vec(train_vocab)

    model = LSTMSentiment(embedding_dim=300, hidden_dim=150, use_gpu=opt.use_gpu, vocab_size=len(train_vocab), batch_size=opt.batch_size, label_size=5)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.NLLLoss()

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    model.embeddings.weight.data.copy_(t.from_numpy(word_vec))

    best_model = model
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii,data_ in tqdm.tqdm








def test(**kwargs):
    pass



if __name__ == "__main__":
    import fire
    fire.Fire()