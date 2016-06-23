import tensorflow as tf
import numpy as np

##########################################################################
#  embedding_lookup + cnn + cosine margine ,  batch
##########################################################################
class InsQACNN(object):
    def __init__(
      self, sequence_length, batch_size,
      vocab_size, embedding_size,
      filter_sizes, num_filters, l2_reg_lambda=0.0):

        #用户问题,字向量使用embedding_lookup
        self.input_x_1 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_1")
        #待匹配正向问题
        self.input_x_2 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_2")
        #负向问题
        self.input_x_3 = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x_3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        print("input_x_1 ", self.input_x_1)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            chars_1 = tf.nn.embedding_lookup(W, self.input_x_1)
            chars_2 = tf.nn.embedding_lookup(W, self.input_x_2)
            chars_3 = tf.nn.embedding_lookup(W, self.input_x_3)
            #self.embedded_chars_1 = tf.nn.dropout(chars_1, self.dropout_keep_prob)
            #self.embedded_chars_2 = tf.nn.dropout(chars_2, self.dropout_keep_prob)
            #self.embedded_chars_3 = tf.nn.dropout(chars_3, self.dropout_keep_prob)
            self.embedded_chars_1 = chars_1
            self.embedded_chars_2 = chars_2
            self.embedded_chars_3 = chars_3
        self.embedded_chars_expanded_1 = tf.expand_dims(self.embedded_chars_1, -1)
        self.embedded_chars_expanded_2 = tf.expand_dims(self.embedded_chars_2, -1)
        self.embedded_chars_expanded_3 = tf.expand_dims(self.embedded_chars_3, -1)

        pooled_outputs_1 = []
        pooled_outputs_2 = []
        pooled_outputs_3 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-1"
                )
                pooled_outputs_1.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_2,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-2"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-2"
                )
                pooled_outputs_2.append(pooled)

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_3,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-3"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="poll-3"
                )
                pooled_outputs_3.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        pooled_reshape_1 = tf.reshape(tf.concat(3, pooled_outputs_1), [-1, num_filters_total]) 
        pooled_reshape_2 = tf.reshape(tf.concat(3, pooled_outputs_2), [-1, num_filters_total]) 
        pooled_reshape_3 = tf.reshape(tf.concat(3, pooled_outputs_3), [-1, num_filters_total]) 
        #dropout
        pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
        pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
        pooled_flat_3 = tf.nn.dropout(pooled_reshape_3, self.dropout_keep_prob)

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_1), 1)) #计算向量长度Batch模式
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_2, pooled_flat_2), 1))
        pooled_len_3 = tf.sqrt(tf.reduce_sum(tf.mul(pooled_flat_3, pooled_flat_3), 1))
        pooled_mul_12 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_2), 1) #计算向量的点乘Batch模式
        pooled_mul_13 = tf.reduce_sum(tf.mul(pooled_flat_1, pooled_flat_3), 1)

        with tf.name_scope("output"):
            self.cos_12 = tf.div(pooled_mul_12, tf.mul(pooled_len_1, pooled_len_2), name="scores") #计算向量夹角Batch模式
            self.cos_13 = tf.div(pooled_mul_13, tf.mul(pooled_len_1, pooled_len_3))

        zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
        margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)
        with tf.name_scope("loss"):
            self.losses = tf.maximum(zero, tf.sub(margin, tf.sub(self.cos_12, self.cos_13)))
            self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
            print('loss ', self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct = tf.equal(zero, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, "float"), name="accuracy")
