#! /usr/bin/env python3.4

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import insurance_qa_data_helpers
from insqa_cnn import InsQACNN
import operator

#print tf.__version__

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
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

vocab = insurance_qa_data_helpers.build_vocab()
alist = insurance_qa_data_helpers.read_alist()
raw = insurance_qa_data_helpers.read_raw()
x_train_1, x_train_2, x_train_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
testList, vectors = insurance_qa_data_helpers.load_test_and_vectors()
vectors = ''
print('x_train_1', np.shape(x_train_1))
print("Load done...")

val_file = '/export/jw/cnn/insuranceQA/test1'
precision = '/export/jw/cnn/insuranceQA/test1.acc'
#x_val, y_val = data_deepqa.load_data_val()

# Training
# ==================================================

with tf.Graph().as_default():
  with tf.device("/gpu:1"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = InsQACNN(
            sequence_length=x_train_1.shape[1],
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
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch_1, x_batch_2, x_batch_3):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x_1: x_batch_1,
              cnn.input_x_2: x_batch_2,
              cnn.input_x_3: x_batch_3,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step():
          scoreList = []
          i = int(0)
          while True:
              x_test_1, x_test_2, x_test_3 = insurance_qa_data_helpers.load_data_val_6(testList, vocab, i, FLAGS.batch_size)
              feed_dict = {
                cnn.input_x_1: x_test_1,
                cnn.input_x_2: x_test_2,
                cnn.input_x_3: x_test_3,
                cnn.dropout_keep_prob: 1.0
              }
              batch_scores = sess.run([cnn.cos_12], feed_dict)
              for score in batch_scores[0]:
                  scoreList.append(score)
              i += FLAGS.batch_size
              if i >= len(testList):
                  break
          sessdict = {}
          index = int(0)
          for line in open(val_file):
              items = line.strip().split(' ')
              qid = items[1].split(':')[1]
              if not qid in sessdict:
                  sessdict[qid] = []
              sessdict[qid].append((scoreList[index], items[0]))
              index += 1
              if index >= len(testList):
                  break
          lev1 = float(0)
          lev0 = float(0)
          of = open(precision, 'a')
          for k, v in sessdict.items():
              v.sort(key=operator.itemgetter(0), reverse=True)
              score, flag = v[0]
              if flag == '1':
                  lev1 += 1
              if flag == '0':
                  lev0 += 1
          of.write('lev1:' + str(lev1) + '\n')
          of.write('lev0:' + str(lev0) + '\n')
          print('lev1 ' + str(lev1))
          print('lev0 ' + str(lev0))
          of.close()

        # Generate batches
        # Training loop. For each batch...
        for i in range(FLAGS.num_epochs):
            try:
                x_batch_1, x_batch_2, x_batch_3 = insurance_qa_data_helpers.load_data_6(vocab, alist, raw, FLAGS.batch_size)
                train_step(x_batch_1, x_batch_2, x_batch_3)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step()
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            except Exception as e:
                print(e)
