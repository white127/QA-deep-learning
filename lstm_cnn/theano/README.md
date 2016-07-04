
theano lstm+cnn code for insuranceQA

================result==================

theano code, test1 top-1 precision : 68.3% 

lstm+cnn is better than cnn(61.5%). 

================dataset================

dataset is large, only test1 sample is given (see ./insuranceQA/test1.sample)

I converted original idx_xx format to real-word format (see ./insuranceQA/train ./insuranceQA/test1.sample)

you can get the original dataset from https://github.com/shuzi/insuranceQA

word embedding is trained by word2vec toolkit

=================run=====================

reformat the original dataset(see my train and test1.sample)

change filepath to your dataset(see TODO in insqa_cnn.py)

python insqa_lstm.py
