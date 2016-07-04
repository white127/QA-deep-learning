
================result==================
theano and tensorflow cnn code for insuranceQA

theano code, test1 top-1 precision : 61.5% (see ./insuranceQA/acc)
tensorflow code, test1 top-1 precision : 62.6%

the best precision in the paper is 62.8% (see Applying Deep Leaarning To Answer Selection: A study and an open task)

================dataset================
dataset is large, only test1 sample is given (see ./insuranceQA/test1.sample)

I converted original idx_xx format to real-word format (see ./insuranceQA/train ./insuranceQA/test1.sample)

you can get the original dataset from https://github.com/shuzi/insuranceQA

word embedding is trained by word2vec toolkit

=================run=====================
reformat the original dataset(see my train and test1.sample)
change filepath to your dataset(see TODO in insqa_cnn.py)
python insqa_cnn.py
