#coding:utf8
import sys,os
import torch as t
from torch import nn
from torch.autograd import Variable
import tqdm
from utils import Visualizer
from torchnet import meter
import ipdb
from models.LSTM import LSTMSentiment
from dataloader import load_train,load_test,get_word_vec
from config import opt


# 将结果写入csv文件
def write_csv(results, file_name):
    import csv
    with open(file_name,'w') as f:
        fieldnames = ['PhraseId', 'Sentiment']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ii in results:
            writer.writerow(ii)


def train_epoch(model, trn, loss_function, optimizer, vis, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    loss_meter = meter.AverageValueMeter()
    loss_meter.reset()

    print("trn.size",trn.__len__())

    for batch in tqdm.tqdm(trn, desc='Train epoch '+str(epoch+1)):
        print("这是第{0}次batch".format(count+1))
        #trn是torchtext的一个batch迭代器
        input, label = batch.text, batch.label
        # 这里是将label的改为从零开始，等着在思考为何要这么做
        # label.data.sub_(1)
        # truth_res += list(label.data)
        model.batch_size = input.size(-1)
        model.hidden = model.init_hidden()

        # input的维度为(X, batch_size),X即一句话的长度，因为train_iter选择的是BucketIterator,它会将相似长度的句子放在一起
        pred = model(input)
        # pred_label = pred.data.max(1)[1].numpy()
        # pred_res += [x for x in pred_label]

        model.zero_grad()
        loss = loss_function(pred, label)
        loss_meter.add(loss.data[0])

        count += 1
        loss.backward()
        optimizer.step()

        if count % opt.plot_every == 0:

            if os.path.exists(opt.debug_file):
                ipdb.set_trace()

            vis.plot('loss', loss_meter.value()[0])

    # avg_loss /= len(trn)
    #acc = get_accuracy(truth_res, pred_res)
    # return avg_loss


# train the model LSTM
def train(**kwargs):
    # fire提供的参数
    for k, v in kwargs.items():
        setattr(opt, k, v)

    vis = Visualizer(env=opt.env)  # 设置visdom的环境变量

    trn, all_vocab = load_train()
    word_vec = get_word_vec(all_vocab)

    model = LSTMSentiment(embedding_dim=300, hidden_dim=150, vocab_size=len(all_vocab), label_size=5)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.NLLLoss()

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    # 将embeddings的数据用word_vec替换
    model.embeddings.weight.data.copy_(t.from_numpy(word_vec))

    for epoch in range(opt.epoch):
        train_epoch(model, trn, criterion, optimizer, vis, epoch)
        t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))


def test(**kwargs):

    for k,v in kwargs.items():
        setattr(opt,k,v)

    _, all_vocab = load_train()
    test_iter = load_test()

    results = []
    #加载模型
    model = LSTMSentiment(embedding_dim=300, hidden_dim=150, vocab_size=len(all_vocab), label_size=5)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    if opt.use_gpu:
        model.cuda()
    model.eval()

    for batch in tqdm.tqdm(test_iter):
        id, input = batch.phrase_id, batch.text
        model.batch_size = input.size(-1)
        model.hidden = model.init_hidden()
        # pred （batch_size,label_size）
        pred = model(input)
        # torch.max(a,1),对行取最大值，0是对列取最大值，[1]是因为max返回两个列表，第二个是argmax，在这里没啥用处
        pred_label = pred.data.max(1)[1].cpu().numpy()
        id = id.data.cpu().numpy()
        # print(id, pred_label)
        result = zip(id, pred_label)
        for x in result:
            tmp = {}
            tmp['PhraseId']=x[0]
            tmp['Sentiment']=x[1]
            results.append(tmp)
    print(results)
    write_csv(results,'results.csv')


if __name__ == "__main__":
    import fire
    fire.Fire()