import numpy as np
import tensorflow as tf
import time, os, random, datetime, sys
from sklearn import metrics
sys.path.append('../')
import config, utils

#top 1 precision:54%
class SWEM_HIER(object):
  def __init__(self, 
      margin,
      sequence_length,
      vocab_size,
      embedding_size,
      embeddings):
    self.zero = tf.placeholder(tf.float32, [None])
    self.q = tf.placeholder(tf.int32, [None, sequence_length])
    self.qp = tf.placeholder(tf.int32, [None, sequence_length])
    self.qn = tf.placeholder(tf.int32, [None, sequence_length])

    with tf.device('/cpu:0'), tf.name_scope('embedding'):
      self.word_mat = tf.Variable(embeddings, trainable=True, dtype=tf.float32)
      q_mat = tf.nn.embedding_lookup(self.word_mat, self.q)
      qp_mat = tf.nn.embedding_lookup(self.word_mat, self.qp)
      qn_mat = tf.nn.embedding_lookup(self.word_mat, self.qn)
      self.q_mat_exp = tf.expand_dims(q_mat, -1)
      self.qp_mat_exp = tf.expand_dims(qp_mat, -1)
      self.qn_mat_exp = tf.expand_dims(qn_mat, -1)

      self.word_mat1 = tf.Variable(embeddings, trainable=True, dtype=tf.float32)
      q_mat1 = tf.nn.embedding_lookup(self.word_mat1, self.q)
      qp_mat1 = tf.nn.embedding_lookup(self.word_mat1, self.qp)
      qn_mat1 = tf.nn.embedding_lookup(self.word_mat1, self.qn)
      self.q_mat_exp1 = tf.expand_dims(q_mat1, -1)
      self.qp_mat_exp1 = tf.expand_dims(qp_mat1, -1)
      self.qn_mat_exp1 = tf.expand_dims(qn_mat1, -1)

    q = tf.nn.avg_pool(self.q_mat_exp, ksize=[1, 2, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    qp = tf.nn.avg_pool(self.qp_mat_exp, ksize=[1, 2, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    qn = tf.nn.avg_pool(self.qn_mat_exp, ksize=[1, 2, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    q = tf.reshape(tf.reduce_max(q, 1), [-1, embedding_size])
    qp = tf.reshape(tf.reduce_max(qp, 1), [-1, embedding_size])
    qn = tf.reshape(tf.reduce_max(qn, 1), [-1, embedding_size])

    q1 = tf.nn.avg_pool(self.q_mat_exp1, ksize=[1, 1, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    qp1 = tf.nn.avg_pool(self.qp_mat_exp1, ksize=[1, 1, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    qn1 = tf.nn.avg_pool(self.qn_mat_exp1, ksize=[1, 1, 1, 1], 
        strides=[1, 1, 1, 1], padding='VALID')
    q1 = tf.reshape(tf.reduce_max(q1, 1), [-1, embedding_size])
    qp1 = tf.reshape(tf.reduce_max(qp1, 1), [-1, embedding_size])
    qn1 = tf.reshape(tf.reduce_max(qn1, 1), [-1, embedding_size])

    q = tf.concat([q, q1], 1)
    qp = tf.concat([qp, qp1], 1)
    qn = tf.concat([qn, qn1], 1)

    self.cos_q_qp = self.cosine(q, qp)
    self.cos_q_qn = self.cosine(q, qn)

    self.losses, loss_batch = self.margin_loss(self.zero, margin, self.cos_q_qp, self.cos_q_qn)

    correct = tf.equal(self.zero, loss_batch)
    self.accuracy = tf.reduce_mean(tf.cast(correct, "float"))

  def margin_loss(self, zero, margin, cos_q_qp, cos_q_qn):
    loss_batch = tf.maximum(zero, tf.subtract(margin, tf.subtract(cos_q_qp, cos_q_qn)))
    losses = tf.reduce_sum(loss_batch)
    return losses, loss_batch

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

margin = 0.05
max_len = 200
num_epoch = 200000
batch_size = 256
checkpoint_every = 50000
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
      swem = SWEM_HIER(margin, max_len, len(vocab), embedding_size, embeddings)
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
        q, qp, qn = utils.gen_train_batch_qpn(train_data, batch_size)
        one, zero = get_constant(batch_size)
        feed_dict = {swem.q:q, swem.qp:qp, swem.qn:qn, swem.zero:zero}
        _, step, loss, cos, acc = sess.run(
            [train_op, global_step, swem.losses, swem.cos_q_qp, swem.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc{:g}".format(time_str, step, loss, acc))

      def test_step():
        yp, y, group = [], [], []
        for i in range(0, len(test_data), batch_size):
          f, g, q1, q2 = utils.gen_test_batch_qpn(test_data, i, i+batch_size)
          one, zero = get_constant(len(f))
          feed_dict = {swem.q:q1, swem.qp:q2, swem.qn:q2, swem.zero:zero}
          loss, cos = sess.run([swem.losses, swem.cos_q_qp], feed_dict)
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
          auc = utils.eval_auc(y, g, yp)
          top1_prec = utils._eval_top1_prec(y, g, yp)
          #if auc < prev_auc:
          #  _flist = [(_f, [s]) for s, _f in zip(score[:len(test_data)], flags)]
          #  features.append(_flist)
          #  break
          #prev_auc = auc

#utils.save_features(features[0] + features[1] + features[2], './data/gen_sweg_hier_train.f')
#utils.save_features(features[3], './data/gen_sweg_hier_test.f')
