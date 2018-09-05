import tensorflow as tf
import numpy as np

##########################################################################
#  embedding_lookup + cnn + cosine margine ,  batch
##########################################################################
class InsQACNN(object):
    def __init__(self, _margin, sequence_length, batch_size,
            vocab_size, embedding_size,
            filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.L, self.B, self.V, self.E, self.FS, self.NF = sequence_length, batch_size, \
                vocab_size, embedding_size, filter_sizes, num_filters 

        #用户问题,字向量使用embedding_lookup
        self.q = tf.placeholder(tf.int32, [self.B, self.L], name="q")
        #待匹配正向问题
        self.qp = tf.placeholder(tf.int32, [self.B, self.L], name="qp")
        #负向问题
        self.qn = tf.placeholder(tf.int32, [self.B, self.L], name="qn")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.get_variable(
                    initializer=tf.random_uniform([self.V, self.E], -1.0, 1.0), 
                    name='We')
            self.qe = tf.nn.embedding_lookup(W, self.q)
            self.qpe = tf.nn.embedding_lookup(W, self.qp)
            self.qne = tf.nn.embedding_lookup(W, self.qn)
        self.qe = tf.expand_dims(self.qe, -1)
        self.qpe = tf.expand_dims(self.qpe, -1)
        self.qne = tf.expand_dims(self.qne, -1)
        
        with tf.variable_scope('shared-conv') as scope:
            self.qe = self.conv(self.qe)
            scope.reuse_variables()
            #tf.get_variable_scope().reuse_variables()
            self.qpe = self.conv(self.qpe)
            scope.reuse_variables()
            #tf.get_variable_scope().reuse_variables()
            self.qne = self.conv(self.qne)
        self.cos_q_qp = self.cosine(self.qe, self.qpe)
        self.cos_q_qn = self.cosine(self.qe, self.qne)
        zero = tf.constant(0, shape=[self.B], dtype=tf.float32)
        margin = tf.constant(_margin, shape=[self.B], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_q_qp, self.cos_q_qn)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")

        for v in tf.trainable_variables():
            print(v)

    def conv(self, tensor):
      pooled = []   
      #with tf.variable_scope(name_or_scope='my-conv', reuse=tf.AUTO_REUSE):
      with tf.variable_scope("my-conv-shared"):
          for i, fs in enumerate(self.FS):
              filter_shape = [fs, self.E, 1, self.NF]
              W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), 
                      name="W-%s" % str(fs))
              b = tf.get_variable(initializer=tf.constant(0.1, shape=[self.NF]), 
                      name="b-%s" % str(fs))
              conv = tf.nn.conv2d(
                      tensor, W, strides=[1, 1, 1, 1], padding='VALID',
                      name="conv")
              h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
              output = tf.nn.max_pool(
                      h, ksize=[1, self.L - fs + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID',
                      name="pool")
              pooled.append(output)
          num_filters_total = self.NF * len(self.FS)
          pooled = tf.reshape(tf.concat(pooled, 3), [-1, num_filters_total])
          pooled = tf.nn.dropout(pooled, self.dropout_keep_prob)
          return pooled

    def cosine(self, v1, v2):
        l1 = tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1))
        l2 = tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1))
        a = tf.reduce_sum(tf.multiply(v1, v2), 1)
        cos = tf.div(a, tf.multiply(l1, l2), name='score')
        return tf.clip_by_value(cos, 1e-5, 0.99999)

