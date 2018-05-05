#coding:utf8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt


class LSTMSentiment(nn.Module):

    '''
    train a LSTM model to classify the text
    '''

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMSentiment, self).__init__()
        self.use_gpu = opt.use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = opt.batch_size
        self.dropout = 0.5
        #随机embeddings,其后可以用word2vec等工具代替
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #2层lstm
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)
        #全连接层,第一个是in_features,第二个是out_feature,这里第二个参数就是label的个数
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


    # 之前的报错说的是hidden的维度不对，要求是(2,60,150),但是得到的确实(2,100,150),这个60不知道哪来的
    def forward(self, input):
        # input （seq_len, batch_size） 每一列是一句话
        # embeds （seq_len, batch_size, embedding_dim）
        # 这里可能batch_size不是预设值，所以要把input的batch_size传
        embeds = self.embeddings(input).view(len(input), self.batch_size, -1)
        # print("input's size", embeds.size())
        # lstm_out （seq_len, batch_size, hidden_dim）
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out[-1] (batch_size,hidden_dim) out （ batch_size,label_size）
        out = self.hidden2lable(lstm_out[-1])
        # 将out转化成概率，lable_scores （batch_size,label_size）
        lable_scores = F.log_softmax(out)
        return lable_scores

