【insuranceQA-cnn-lstm README】

See theano and tensorflow folder

This is a CNN/RNN model for Q&A(Question and Answering), include theano and tensorflow code implementation

【insuranceQA准确率】

CNN+MaxPooling top1准确率 62% insuranceQA-cnn-lstm/cnn/tensorflow/

LSTM+CNN+MaxPooling top1准确率 68% insuranceQA-cnn-lstm/lstm_cnn/theano/

GRU+MaxPooling top1准确率 59% insuranceQA-cnn-lstm/rnn_attention/tensorflow/

【insuranceQA数据说明】

数据下载，原始数据-https://github.com/shuzi/insuranceQA ，需要进行处理转换之后才能在这里的代码中使用

转换后的数据格式参照这里：
https://github.com/white127/insuranceQA-cnn-lstm/tree/master/insuranceQA/test1.sample

数据格式转换使用 gen.py(需要更改代码中的文件路径)，生成模型的输入文件

【文本分类】

使用CNN和RNN进行文本分类的代码请移步 https://github.com/white127/TextClassification_CNN_RNN
