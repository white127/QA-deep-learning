import numpy as np
import tensorflow as tf
import time, os, random, datetime, sys
from sklearn import metrics
sys.path.append('../')
import config, utils

################################################################################
# Insurance-QA
# AUC 0.96, top 1 precision:31%
#
# quora-data
# best precision: 0.8369, best threshold:0.62
################################################################################
class SWEM_HIER(object):
  def __init__(self, 
      sequence_length,
      vocab_size,
      embedding_size,
      embeddings):
    self.x1 = tf.placeholder(tf.int32, [None, sequence_length])
    self.x2 = tf.placeholder(tf.int32, [None, sequence_length])
    self.y = tf.placeholder(tf.float32, [None])
    self.one = tf.placeholder(tf.float32, [None])
    #self.dropout_keep_prob = tf.placeholder(tf.float32)

    with tf.device('/cpu:0'), tf.name_scope('embedding'):
      self.word_mat = tf.Variable(embeddings, trainable=True, dtype=tf.float32)
      x1_mat = tf.nn.embedding_lookup(self.word_mat, self.x1)
      x2_mat = tf.nn.embedding_lookup(self.word_mat, self.x2)
      self.x1_mat_exp = tf.expand_dims(x1_mat, -1)
      self.x2_mat_exp = tf.expand_dims(x2_mat, -1)
    p1 = tf.nn.avg_pool(self.x1_mat_exp, ksize=[1, 2, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    p2 = tf.nn.avg_pool(self.x2_mat_exp, ksize=[1, 2, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    p1 = tf.reshape(tf.reduce_max(p1, 1), [-1, embedding_size])
    p2 = tf.reshape(tf.reduce_max(p2, 1), [-1, embedding_size])
    """
    p11 = tf.nn.avg_pool(self.x1_mat_exp, ksize=[1, 3, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    p21 = tf.nn.avg_pool(self.x2_mat_exp, ksize=[1, 3, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    p11 = tf.reshape(tf.reduce_max(p11, 1), [-1, embedding_size])
    p21 = tf.reshape(tf.reduce_max(p21, 1), [-1, embedding_size])
    p1 = tf.concat([p1, p11], 1)
    p2 = tf.concat([p2, p21], 1)
    """

    self.cos = self.cosine(p1, p2)
    self.losses = self.logloss(self.y, self.one, self.cos)

  def logloss(self, y, v_one, sim):
    a = tf.multiply(y, tf.log(sim)) #y*log(p)
    b = tf.subtract(v_one, y)#1-y
    c = tf.log(tf.subtract(v_one, sim))#log(1-p)
    losses = -tf.add(a, tf.multiply(b, c))#y*log(p)+(1-y)*log(1-p)
    losses = tf.reduce_sum(losses, -1)
    return losses

  def cosine(self, t1, t2):
    len1 = tf.sqrt(tf.reduce_sum(tf.multiply(t1, t1), 1))
    len2 = tf.sqrt(tf.reduce_sum(tf.multiply(t2, t2), 1))
    multiply = tf.reduce_sum(tf.multiply(t1, t2), 1)
    cos = tf.div(multiply, tf.multiply(len1, len2))
    return tf.clip_by_value(cos, 1e-5, 0.99999)

def get_constant(batch_size):
  one, zero = [1.0] * batch_size, [0.0] * batch_size
  return np.array(one), np.array(zero)

max_len = 100
num_epoch = 200000
batch_size = 256
checkpoint_every = 10000
vocab, embeddings = utils.load_embeddings()
embedding_size = len(embeddings[0])
train_data, test_data = utils.load_train_data(vocab, max_len), utils.load_test_data(vocab, max_len)
print('load data done ......')
print(embeddings.shape)

prev_auc = 0.0
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      swem = SWEM_HIER(max_len, len(vocab), embedding_size, embeddings)
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(1e-1)
      #optimizer = tf.train.GradientDescentOptimizer(1e-1)
      grads_and_vars = optimizer.compute_gradients(swem.losses)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

      timestamp = str(int(time.time()))
      out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
      checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
      checkpoint_prefix = os.path.join(checkpoint_dir, "model")
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
      saver = tf.train.Saver(tf.all_variables())
      sess.run(tf.initialize_all_variables())

      def train_step():
        y, x1, x2 = utils.gen_train_batch_yxx(train_data, batch_size)
        one, zero = get_constant(batch_size)
        feed_dict = {swem.x1:x1, swem.x2:x2, swem.y:y, swem.one:one}
        _, step, loss, cos = sess.run(
            [train_op, global_step, swem.losses, swem.cos], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))

      def test_step():
        yp, y, group = [], [], []
        for i in range(0, len(test_data), batch_size):
          f, g, x1, x2 = utils.gen_test_batch_yxx(test_data, i, i + batch_size)
          one, zero = get_constant(len(f))
          feed_dict = {swem.x1:x1, swem.x2:x2, swem.y:f, swem.one:one}
          loss, cos = sess.run([swem.losses, swem.cos], feed_dict)
          yp.extend(cos)
          y.extend(f)
          group.extend(g)
        ppp = [(_y, _g, _yp) for _y, _g, _yp in zip(y, group, yp)]
        #for _y, _g, _yp in ppp:
        #  print(str(_y) + ' ' + str(_g) + ' ' + str(_yp))
        return y[:len(test_data)], group[:len(test_data)], yp[:len(test_data)]

      for i in range(num_epoch):
        train_step()
        current_step = tf.train.global_step(sess, global_step)
        if current_step % checkpoint_every == 0:
          y, g, yp = test_step()
          utils._eval(y, g, yp)

#utils.save_features(features[0] + features[1] + features[2], './data/gen_sweg_hier_train.f')
#utils.save_features(features[3], './data/gen_sweg_hier_test.f')
