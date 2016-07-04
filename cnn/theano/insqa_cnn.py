
###########################################################
# test1 top-1 precision: 62%
###########################################################

import os, sys, timeit, random, operator

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

#TODO change path to your dataset
trainfile = '/export/jw/cnn/insuranceQA/train'
test1file = '/export/jw/cnn/insuranceQA/test1'
vectorsfile = '/export/jw/cnn/insuranceQA/vectors.nobin'

###########################################################
# read qa data
###########################################################
def build_vocab():
    global trainfile
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(trainfile):
        items = line.strip().split(' ')
        for i in range(2, 3):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab

def load_vectors():
    global vectorsfile
    vectors = {}
    for line in open(vectorsfile):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_vectors()
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, dim):
            vec.append(0.01)
        embeddings.append(vec)
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
    return np.array(embeddings, dtype='float32')

#be attention initialization of UNKNNOW
def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x

def load_train_list():
    global trainfile
    trainList = []
    for line in open(trainfile):
        trainList.append(line.strip().split(' '))
    return trainList

def load_test_list():
    global test1file
    testList = []
    for line in open(test1file):
        testList.append(line.strip().split(' '))
    return testList

def load_data(trainList, vocab, batch_size):
    train_1, train_2, train_3 = [], [], []
    for i in range(0, batch_size):
        pos = trainList[random.randint(0, len(trainList)-1)]
        neg = trainList[random.randint(0, len(trainList)-1)]
        train_1.append(encode_sent(vocab, pos[2], 100))
        train_2.append(encode_sent(vocab, pos[3], 100))
        train_3.append(encode_sent(vocab, neg[3], 100))
    return np.array(train_1, dtype='float32'), np.array(train_2, dtype='float32'), np.array(train_3, dtype='float32')

def load_data_val(testList, vocab, index, batch_size):
    x1, x2, x3 = [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        items = testList[true_index]
        x1.append(encode_sent(vocab, items[2], 100))
        x2.append(encode_sent(vocab, items[3], 100))
        x3.append(encode_sent(vocab, items[3], 100))
    return np.array(x1, dtype='float32'), np.array(x2, dtype='float32'), np.array(x3, dtype='float32')

def validation(validate_model, testList, vocab, batch_size):
    index, score_list = int(0), []
    while True:
        x1, x2, x3 = load_data_val(testList, vocab, index, batch_size)
        batch_scores, nouse = validate_model(x1, x2, x3, 1.0)
        for score in batch_scores:
            score_list.append(score)
        index += batch_size
        if index >= len(testList):
            break
        print 'Evaluation ' + str(index)
    sdict, index = {}, int(0)
    for items in testList:
        qid = items[1].split(':')[1]
        if not qid in sdict:
            sdict[qid] = []
        sdict[qid].append((score_list[index], items[0]))
        index += 1
    lev0, lev1 = float(0), float(0)
    for qid, cases in sdict.items():
        cases.sort(key=operator.itemgetter(0), reverse=True)
        score, flag = cases[0]
        if flag == '1':
            lev1 += 1
        if flag == '0':
            lev0 += 1
    print 'top-1 precition: ' + str(lev1 / (lev0 + lev1))

class QACnn(object):
  def __init__(self, input1, input2, input3, word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters, keep_prob):
    rng = np.random.RandomState(23455)
    self.params = []

    lookup_table = theano.shared(word_embeddings)
    self.params += [lookup_table]
    #input1-问题, input2-正向答案, input3-负向答案
    #将每个字替换成字向量
    input_matrix1 = lookup_table[T.cast(input1.flatten(), dtype="int32")]
    input_matrix2 = lookup_table[T.cast(input2.flatten(), dtype="int32")]
    input_matrix3 = lookup_table[T.cast(input3.flatten(), dtype="int32")]

    #CNN的输入是4维矩阵，这里只是增加了一个维度而已
    input_x1 = input_matrix1.reshape((batch_size, 1, sequence_len, embedding_size))
    input_x2 = input_matrix2.reshape((batch_size, 1, sequence_len, embedding_size))
    input_x3 = input_matrix3.reshape((batch_size, 1, sequence_len, embedding_size))
    #print(input_x1.shape.eval())
    self.dbg_x1 = input_x1

    outputs_1, outputs_2, outputs_3 = [], [], []
    #设置多种大小的filter
    for filter_size in filter_sizes:
        #每种大小的filter的数量是num_filters
        filter_shape = (num_filters, 1, filter_size, embedding_size)
        image_shape = (batch_size, 1, sequence_len, embedding_size)
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)

        #卷积+max_pooling
        conv_out = conv2d(input=input_x1, filters=W, filter_shape=filter_shape, input_shape=image_shape)
        #卷积后的向量的长度为ds
        pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
        outputs_1.append(pooled_active)

        conv_out = conv2d(input=input_x2, filters=W, filter_shape=filter_shape, input_shape=image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
        outputs_2.append(pooled_active)

        conv_out = conv2d(input=input_x3, filters=W, filter_shape=filter_shape, input_shape=image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
        outputs_3.append(pooled_active)

        self.params += [W, b]
        self.dbg_conv_out = conv_out.shape

    num_filters_total = num_filters * len(filter_sizes)
    self.dbg_outputs_1 = outputs_1[0].shape
    #每一个句子的语义表示向量的长度为num_filters_total
    output_flat1 = T.reshape(T.concatenate(outputs_1, axis=1), [batch_size, num_filters_total])
    output_flat2 = T.reshape(T.concatenate(outputs_2, axis=1), [batch_size, num_filters_total])
    output_flat3 = T.reshape(T.concatenate(outputs_3, axis=1), [batch_size, num_filters_total])
    #dropout, keep_prob为1表示不进行dropout
    output_drop1 = self._dropout(rng, output_flat1, keep_prob)
    output_drop2 = self._dropout(rng, output_flat2, keep_prob)
    output_drop3 = self._dropout(rng, output_flat3, keep_prob)

    #计算问题和答案之前的向量夹角
    #计算向量的长度
    len1 = T.sqrt(T.sum(output_drop1 * output_drop1, axis=1))
    len2 = T.sqrt(T.sum(output_drop2 * output_drop2, axis=1))
    len3 = T.sqrt(T.sum(output_drop3 * output_drop3, axis=1))
    #计算向量之间的夹角
    cos12 = T.sum(output_drop1 * output_drop2, axis=1) / (len1 * len2)
    self.cos12 = cos12
    cos13 = T.sum(output_drop1 * output_drop3, axis=1) / (len1 * len3)
    self.cos13 = cos13

    zero = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX), borrow=True)
    margin = theano.shared(np.full(batch_size, 0.05, dtype=theano.config.floatX), borrow=True)
    #Loss损失函数
    diff = T.cast(T.maximum(zero, margin - cos12 + cos13), dtype=theano.config.floatX)
    self.cost = T.sum(diff, acc_dtype=theano.config.floatX)
    #mini-batch数据的准确率(如果正向答案和问题之间的cosine大于负向答案和问题的cosine，则认为正确，
    #否则是错误的)
    #Loss和Accuracy是用来评估训练中模型时候收敛的两个很重要的指标
    self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)

  def _dropout(self, rng, layer, keep_prob):
    srng = T.shared_randomstreams.RandomStreams(rng.randint(123456))
    mask = srng.binomial(n=1, p=keep_prob, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    output = output / keep_prob
    return output

def train():
    batch_size = int(256)
    filter_sizes = [2,3,5]
    num_filters = 500
    embedding_size = 100
    learning_rate = 0.001
    n_epochs = 2000000
    validation_freq = 1000
    keep_prob_value = 0.25

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    testList = load_test_list()
    train_x1, train_x2, train_x3 = load_data(trainList, vocab, batch_size)

    x1, x2, x3 = T.matrix('x1'), T.matrix('x2'), T.matrix('x3')
    keep_prob = T.fscalar('keep_prob')
    model = QACnn(
        input1=x1, input2=x2, input3=x3, keep_prob=keep_prob,
        word_embeddings=word_embeddings, 
        batch_size=batch_size,
        sequence_len=train_x1.shape[1],
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters)
    dbg_x1 = model.dbg_x1
    dbg_outputs_1 = model.dbg_outputs_1

    cost, cos12, cos13 = model.cost, model.cos12, model.cos13
    print 'cost'
    print cost
    params, accuracy = model.params, model.accuracy
    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    p1, p2, p3 = T.matrix('p1'), T.matrix('p2'), T.matrix('p3')
    prob = T.fscalar('prob')
    train_model = theano.function(
        [p1, p2, p3, prob], 
        [cost, accuracy, dbg_x1, dbg_outputs_1], 
        updates=updates,
        givens={
            x1: p1, x2: p2, x3: p3, keep_prob: prob
        }
    )

    v1, v2, v3 = T.matrix('v1'), T.matrix('v2'), T.matrix('v3')
    validate_model = theano.function(
        inputs=[v1, v2, v3, prob],
        outputs=[cos12, cos13],
        #updates=updates,
        givens={
            x1: v1, x2: v2, x3: v3, keep_prob: prob
        }
    )

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_x1, train_x2, train_x3 = load_data(trainList, vocab, batch_size)
        #print train_x3.shape
        cost_ij, acc, dbg_x1, dbg_outputs_1 = train_model(train_x1, train_x2, train_x3, keep_prob_value)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', acc:' + str(acc)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            validation(validate_model, testList, vocab, batch_size)
        #print dbg_outputs_1

if __name__ == '__main__':
    train()
