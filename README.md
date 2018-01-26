【insuranceQA-cnn-lstm README】

See theano and tensorflow folder

This is a CNN/LSTM model for Q&A(Question and Answering), include theano and tensorflow code implementation

theano和tensorflow的网络结构都是一致的:
word embedings + CNN + max pooling + cosine similarity

目前再insuranceQA的test1数据集上，top-1准确率可以达到62%左右，跟论文上是一致的。

这里只提供了CNN的代码，后面我测试了LSTM和LSTM+CNN的方法，LSTM+CNN的方法比单纯使用CNN或LSTM效果还要更好一些，在test1上的准确率可以再提示5%-6%

LSTM+CNN的方法在insuranceQA的test1上的准确率为68%

很多人都在问数据下载，这里有原始数据-https://github.com/shuzi/insuranceQA ，但是需要进行处理转换之后才能在这里的代码中使用

转换后的数据格式在这里：
https://github.com/white127/insuranceQA-cnn-lstm/tree/master/insuranceQA/test1.sample

【code of Text Classification】

使用CNN和RNN进行文本分类的代码请移步这里 https://github.com/white127/TextClassification_CNN_RNN
