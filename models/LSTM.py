#coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTMSentiment(nn.Module):

    '''
    train a LSTM model to classify the text
    '''

    def __init__(self, embedding_dim, hidden_dim,vocab_size, batch_size ,label_size, use_gpu):
        super(LSTMSentiment, self).__init__()
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        #随机embeddings,其后可以用word2vec等工具代替
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        #2层lstm
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim,num_layers=2)
        #全连接层
        self.hidden2lable = nn.Linear(self.hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        #Define the hidden state and the cell, the parameter 2's meaning is two layers of LSTM.
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self,input):
        embeds = self.embeddings(input).view(len(input), self.batch_size, -1)
        lstm_out,self.hidden = self.lstm(embeds, self.hidden)
        out = self.hidden2lable(lstm_out[-1])
        lable_scores = F.log_softmax(out)
        return lable_scores

