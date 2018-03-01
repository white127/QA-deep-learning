# -*- coding: utf-8 -*-

####################################################################################
#test1 top1准确率59%
####################################################################################
import tensorflow as tf
import numpy as np
from operator import itemgetter
import random, datetime, json, insurance_qa_data_helpers

class RNN_Model(object):
    def _rnn_net(self, inputs, mask, embedding, keep_prob, batch_size, embed_dim, num_step, fw_cell, bw_cell):
        _initial_state = fw_cell.zero_state(batch_size,dtype=tf.float32)
        inputs=tf.nn.embedding_lookup(embedding, inputs)
        inputs = tf.nn.dropout(inputs, self.keep_prob)
        #[batch_size, sequence_length, embedding_size]转换为[sequence_length, batch_size, embedding_size]
        inputs = tf.transpose(inputs, [1, 0, 2])
        #[sequence_length, batch_size, embedding_size]转换为list, sequence_length个[batch_size, embedding_size]
        inputs = tf.unstack(inputs)
        #inputs = tf.reshape(inputs, [-1, embed_dim])
        #inputs = tf.split(inputs, num_step, 0)
        #输出为list, sequence_length个[batch_size, embedding_size * 2]
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputs, initial_state_fw=_initial_state, initial_state_bw=_initial_state)
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
        self.outputs = outputs
        #对rnn的输出[batch_size, sequence_length, embedding_size],目前采用maxpooling是最好的效果
        #mean_pooling以及取最后一个step的向量,效果都不好
        outputs = self._max_pooling(outputs)
        print outputs
        
        #outputs = outputs[-1]
        #outputs = outputs * mask[:, :, None]
        #mean pooling
        #outputs = tf.reduce_sum(outputs, 0) / (tf.reduce_sum(mask, 0)[:,None])
        return outputs

    def _max_pooling(self, lstm):
        sequence_length, embedding_size = int(lstm.get_shape()[1]), int(lstm.get_shape()[2])
        lstm = tf.expand_dims(lstm, -1)
        output = tf.nn.max_pool(lstm, ksize=[1, sequence_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        output = tf.reshape(output, [-1, embedding_size])
        return output
        
    def __init__(self, config, is_training=True):
        self.keep_prob=tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size=config.batch_size
        self.num_step=config.num_step

        self.qlist = tf.placeholder(tf.int32, [self.batch_size, self.num_step])
        #这个版本没有使用mask
        self.mask_q = tf.placeholder(tf.float32, [self.num_step, self.batch_size])
        self.plist = tf.placeholder(tf.int32, [self.batch_size, self.num_step])
        self.mask_p = tf.placeholder(tf.float32, [self.num_step, self.batch_size])
        self.nlist = tf.placeholder(tf.int32, [self.batch_size, self.num_step])
        self.mask_n = tf.placeholder(tf.float32, [self.num_step, self.batch_size])

        hidden_neural_size=config.hidden_neural_size
        vocabulary_size=config.vocabulary_size
        self.embed_dim=config.embed_dim
        hidden_layer_num=config.hidden_layer_num

        #fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=1.0,state_is_tuple=True)
        fw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_neural_size, activation=tf.nn.relu)
        fw_cell =  tf.contrib.rnn.DropoutWrapper(
            fw_cell,output_keep_prob=self.keep_prob
        )
        #bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=1.0,state_is_tuple=True)
        bw_cell = tf.contrib.rnn.GRUCell(num_units=hidden_neural_size, activation=tf.nn.relu)
        bw_cell =  tf.contrib.rnn.DropoutWrapper(
            bw_cell,output_keep_prob=self.keep_prob
        )

        #embedding layer
        with tf.device("/cpu:1"),tf.name_scope("embedding_layer"):
            self.embedding = tf.Variable(tf.truncated_normal([vocabulary_size, self.embed_dim], stddev=0.1), name='W')
            #self.a_embedding = tf.Variable(tf.truncated_normal([vocabulary_size, self.embed_dim], stddev=0.1), name='W')

        q = self._rnn_net(self.qlist, mask_q, self.embedding, self.keep_prob, self.batch_size, self.embed_dim, self.num_step, fw_cell, bw_cell)
        tf.get_variable_scope().reuse_variables()
        p = self._rnn_net(self.plist, mask_p, self.embedding, self.keep_prob, self.batch_size, self.embed_dim, self.num_step, fw_cell, bw_cell)
        tf.get_variable_scope().reuse_variables()
        n = self._rnn_net(self.nlist, mask_n, self.embedding, self.keep_prob, self.batch_size, self.embed_dim, self.num_step, fw_cell, bw_cell)
        #len_1 = tf.clip_by_value(tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1)), 0.01, 100000)
        #len_2 = tf.clip_by_value(tf.sqrt(tf.reduce_sum(tf.multiply(p, p), 1)), 0.01, 100000)
        #len_3 = tf.clip_by_value(tf.sqrt(tf.reduce_sum(tf.multiply(n, n), 1)), 0.01, 100000)
        len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(p, p), 1))
        len_3 = tf.sqrt(tf.reduce_sum(tf.multiply(n, n), 1))

        self.cos12 = tf.reduce_sum(tf.multiply(q, p), axis=1) / (len_1 * len_2)
        self.cos13 = tf.reduce_sum(tf.multiply(q, n), axis=1) / (len_1 * len_3)
        self.q = q
        self.p = p

        zero = tf.constant(np.zeros(self.batch_size, dtype='float32'))
        margin = tf.constant(np.full(self.batch_size, 0.1, dtype='float32'))
        diff = tf.cast(tf.maximum(zero, margin - self.cos12 + self.cos13), dtype='float32')
        self.cost = tf.reduce_sum(diff)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(zero, diff), dtype='float32')) / float(self.batch_size)

def train_step(model, qlist, plist, nlist, mask_q, mask_p, mask_n):
    fetches = [model.cost, model.accuracy, global_step, train_op, model.cos12, model.q, model.p, model.outputs]
    feed_dict = {
        model.qlist: qlist,
        model.plist: plist,
        model.nlist: nlist,
        model.mask_q : mask_q,
        model.mask_p : mask_p,
        model.mask_n : mask_n,
        model.keep_prob: config.keep_prob
    }
    cost, accuracy, step, _, cos12, q, p, outputs = sess.run(fetches, feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))


def dev_step(model, vocab, batch_size, max_len):
    score_list, i = [], int(0)
    while True:
        qlist, plist, nlist, mask_q, mask_p, mask_n = insurance_qa_data_helpers.load_val_data(test_list, vocab, i, FLAGS.batch_size, max_len)
        feed_dict = {
            model.qlist: qlist,
            model.plist: plist,
            model.nlist: nlist,
            model.mask_q : mask_q,
            model.mask_p : mask_p,
            model.mask_n : mask_n,
            model.keep_prob: float(1.0)
        }
        batch_scores = sess.run([model.cos12], feed_dict)
        for score in batch_scores[0]:
            score_list.append(score)
        i += FLAGS.batch_size
        if i >= len(test_list):
            break
    insurance_qa_data_helpers.evaluation(score_list, test_list)

tf.flags.DEFINE_integer('evaluate_every',10000,'evaluate every')
tf.flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
tf.flags.DEFINE_integer('emdedding_dim',100,'embedding dim')
tf.flags.DEFINE_integer('hidden_neural_size',200,'LSTM hidden neural size')
tf.flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
tf.flags.DEFINE_integer('max_len',100,'max_len of training sentence')
tf.flags.DEFINE_float('init_scale',0.1,'init scale')
tf.flags.DEFINE_float('keep_prob',0.5,'dropout rate')
tf.flags.DEFINE_integer('num_epoch',1000000,'num epoch')
tf.flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

vocab = insurance_qa_data_helpers.build_vocab()
train_list = insurance_qa_data_helpers.load_train_list()
qlist, plist, nlist, mask_q, mask_p, mask_n = insurance_qa_data_helpers.load_train_data(train_list, vocab, FLAGS.batch_size, FLAGS.max_len)
test_list = insurance_qa_data_helpers.load_test_list()

class Config(object):
    hidden_neural_size=FLAGS.hidden_neural_size
    vocabulary_size=len(vocab)
    embed_dim=FLAGS.emdedding_dim
    hidden_layer_num=FLAGS.hidden_layer_num
    keep_prob=FLAGS.keep_prob
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm=FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch

config = Config()
eval_config=Config()
eval_config.keep_prob=1.0

with tf.Graph().as_default():
    with tf.device('/gpu:1'):
      session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,1*FLAGS.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            model = RNN_Model(config=config, is_training=True)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #optimizer = tf.train.RMSPropOptimizer(0.01)
        #optimizer = tf.train.AdamOptimizer(0.1)
        optimizer = tf.train.GradientDescentOptimizer(0.2)
        grads_and_vars = optimizer.compute_gradients(model.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        for i in range(config.num_epoch):
            qlist, plist, nlist, mask_q, mask_p, mask_n = insurance_qa_data_helpers.load_train_data(train_list, vocab, FLAGS.batch_size, FLAGS.max_len)
            train_step(model, qlist, plist, nlist, mask_q, mask_p, mask_n)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_step(model, vocab, FLAGS.batch_size, FLAGS.max_len)
