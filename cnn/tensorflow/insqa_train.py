#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os, time, datetime, operator, sys
from insqa_cnn import InsQACNN
sys.path.append('../../')
import config, utils

print(tf.__version__)

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_float("margin", 0.05, "CNN model margin")
tf.flags.DEFINE_integer("sequence_length", 200, "Max sequence lehgth(default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 3000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
vocab, embeddings = utils.load_embeddings()
train_data = utils.load_train_data(vocab, FLAGS.sequence_length)
test_data = utils.load_test_data(vocab, FLAGS.sequence_length)
print("Load done...")

# Training
# ==================================================

prev_auc = 0
with tf.Graph().as_default():
  with tf.device("/gpu:1"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = InsQACNN(
            _margin=FLAGS.margin,
            sequence_length=FLAGS.sequence_length,
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        #optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(q, qp, qn):
            feed_dict = {
              cnn.q: q, cnn.qp: qp, cnn.qn: qn,
              #cnn.input_x_1: q, cnn.input_x_2: qp, cnn.input_x_3: qn,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, cos1, cos2 = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.cos_q_qp, cnn.cos_q_qn],
                feed_dict)
            #print(cos1)
            #print(cos2)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def test_step():
          yp, y, group, of = [], [], [], open(config.predict1_file, 'w')
          for i in range(0, len(test_data), FLAGS.batch_size):
              f, g, q1, q2 = utils.gen_test_batch_qpn(test_data, i, i+FLAGS.batch_size)
              feed_dict = {
                cnn.q: q1, cnn.qp: q2, cnn.qn: q2,
                #cnn.input_x_1: q1, cnn.input_x_2: q2, cnn.input_x_3: q2,
                cnn.dropout_keep_prob: 1.0
              }
              cos = sess.run([cnn.cos_q_qp], feed_dict)
              yp.extend(cos[0])
              y.extend(f)
              group.extend(g)
          y, g, yp = y[:len(test_data)], group[:len(test_data)], yp[:len(test_data)]
          auc = utils.eval_auc(y[:len(test_data)], g, yp[:len(test_data)])
          top1_prec = utils._eval_top1_prec(y, g, yp)
          for p in yp[:len(test_data)]: of.write(str(p) + '\n')
          of.write(str(top1_prec) + '\n')
          of.close()
          return auc

        # Generate batches
        # Training loop. For each batch...
        for i in range(FLAGS.num_epochs):
            try:
                q, qp, qn = utils.gen_train_batch_qpn(train_data, FLAGS.batch_size)
                train_step(q, qp, qn)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    auc = test_step()
                    #if auc < prev_auc: break
                    prev_auc = auc
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            except Exception as e:
                print(e)
