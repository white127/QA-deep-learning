
================result==================

结果和theano版本的差不多，具体数值忘了

虽然代码里写了dropout，但是实际并没有使用，dropout对结果影响不是特别大，不用dropout的话训练速度要快一些。

================dataset================

数据格式和theano版本的是一样的

github上给出的是样本数据，如果需要全量的，也可直接联系我
dataset is large, only test1 sample is given (see ./insuranceQA/test1.sample)

I converted original idx_xx format to real-word format (see ./insuranceQA/train ./insuranceQA/test1.sample)

you can get the original dataset from https://github.com/shuzi/insuranceQA

word embedding is trained by word2vec toolkit

=================run=====================

./insqa_train.py

我使用的是python3.4，部分代码可能会和python2不兼容，如使用python2需要自己做一些小修改，核心的CNN代码应该
不用改动的
代码里的数据路径(类似'/export/...')是需要根据自己的环境修改的，指向自己的数据路径即可。核心的CNN代码无需改动
