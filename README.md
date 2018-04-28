# 基于词嵌入的文本分类


## 各文件的作用

name | function
:----: | :-----:
main.py| 调用模型进行训练
model.py| CNN或RNN模型
data.py | 加载数据
utils.py | 可视化工具


## 数据集
采用kaggle的数据[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

## 模型

实现Continuous BOW模型、CNN、RNN的文本分类，并进行效果对比。


词用embedding的方式初始化:

- 随机embedding的方式
- 使用word2vec等工具训练出来的文本进行初始化

暂时先使用pytorch的nn.embedding函数，整个框架搭起来后再进行扩展。

## 优化策略

后期可以研究一下如何处理停用词等元素，如何进行预处理可以提高准确率。
