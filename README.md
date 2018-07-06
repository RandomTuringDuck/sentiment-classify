# 基于词嵌入的文本分类

+ 使用pytorch进行完成，python依赖都在requirement.txt中。
+ 由于GoogleNews-vectors-negative300.bin太大，未上传，如需要自行下载放入data文件夹中。
+ 训练后的模型存入了checkpoints文件夹中。


## 训练和测试命令
### 训练
> python main.py train --batch-size=100 --plot-every=200

### 测试
> python main.py test --model-path="checkpoints/_9.pth"

测试结果存入submission.csv文件中。该任务是一个情感分类任务，共分为五类。

## 各文件的作用

name | function
:----: | :-----:
main.py| 调用模型进行训练
model.py| CNN或RNN模型
data.py | 加载数据
utils.py | 可视化工具

这里只使用了torch自带的LSTM进行了简单的训练。


## 数据集
采用kaggle的数据[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

## 模型

实现Continuous BOW模型、CNN、RNN的文本分类，并进行效果对比。


词用embedding的方式初始化:

- 随机embedding的方式
- 使用word2vec等工具训练出来的文本进行初始化

使用google训练好的GoogleNews-vectors-negative300来进行初始化，并将weight赋给
nn.embedding使用。

## 优化策略

没有对停用词进行处理，未对词干进行处理，一些没用的
而且是在word级别上进行的分类，可以尝试phrase级别。

## 防止过拟合
没有使用防止过拟合的策略，仅仅跑了几个epoch而已，准确率不会太高。
