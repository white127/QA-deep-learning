import numpy as np
import random, sys, config
from sklearn import metrics
from operator import itemgetter
from itertools import groupby

def load_embeddings():
  _data, embeddings, vocab, _id = [], [], {}, int(0)
  for line in open(config.w2v_bin_file):
    _data.append(line.strip().split(' '))
  size, dim = int(_data[0][0]), int(_data[0][1])
  for i in range(1, len(_data)):
    w, vec = _data[i][0], [float(_data[i][k]) for k in range(1, dim+1)]
    embeddings.append(vec)
    vocab[w] = _id
    _id += 1
  embeddings.append([0.01] * dim)
  vocab['UNKNOWN'] = _id
  _id += 1
  embeddings.append([0.01] * dim)
  vocab['<a>'] = _id
  return vocab, np.array(embeddings)

def encode_sent(s, vocab, max_len):
  ws = [w for w in s.split('_')]
  ws = ws[:max_len] if len(ws) >= max_len else ws + ['<a>'] * (max_len - len(ws)) 
  nws = []
  for w in ws:
    nw = w if w in vocab else 'UNKNOWN'
    nws.append(vocab[nw])
  return nws

def load_train_data(vocab, max_len):
  if config.dataset == config.dataset_ins:
    return ins_load_train_data(vocab, max_len)
  if config.dataset == config.dataset_qur:
    return qur_load_train_test_data(config.train_file, vocab, max_len)
  print('bad load_train_data')
  exit(1)

def qur_load_train_test_data(_file, vocab, max_len):
  _data = []
  for line in open(_file):
    f, q1, q2 = line.strip().split(' ')
    q1, q2 = encode_sent(q1, vocab, max_len), encode_sent(q2, vocab, max_len)
    _data.append((int(f), q1, q2))
  return _data

def ins_load_train_data(vocab, max_len):
  _data = []
  for line in open(config.train_file):
    f, q1, q2 = line.strip().split(' ')
    q1, q2 = encode_sent(q1, vocab, max_len), encode_sent(q2, vocab, max_len)
    _data.append((q1, q2))
  return _data

def load_test_data(vocab, max_len):
  if config.dataset == config.dataset_ins:
    return ins_load_test_data(vocab, max_len)
  if config.dataset == config.dataset_qur:
    return qur_load_train_test_data(config.test1_file, vocab, max_len)
  print('bad load_test_data')
  exit(1)

def ins_load_test_data(vocab, max_len):
  _data = []
  for line in open(config.test1_file):
    f, g, q1, q2 = line.strip().split(' ')
    q1, q2 = encode_sent(q1, vocab, max_len), encode_sent(q2, vocab, max_len)
    _data.append((f, g, q1, q2))
  return _data

def gen_train_batch_qpn(_data, batch_size):
  psample = random.sample(_data, batch_size)
  nsample = random.sample(_data, batch_size)
  q = [s1 for s1, s2 in psample]
  qp = [s2 for s1, s2 in psample]
  qn = [s2 for s1, s2 in nsample]
  return np.array(q), np.array(qp), np.array(qn)

def gen_train_batch_yxx(_data, batch_size):
  if config.dataset == config.dataset_ins:
    return ins_gen_train_batch_yxx(_data, batch_size)
  if config.dataset == config.dataset_qur:
    return qur_gen_train_batch_yxx(_data, batch_size)
  print('bad gen_train_batch_yxx')
  exit(1)

def qur_gen_train_batch_yxx(_data, batch_size):
  sample = random.sample(_data, batch_size)
  y = [i for i,_,_ in sample]
  x1 = [i for _,i,_ in sample]
  x2 = [i for _,_,i in sample]
  return np.array(y), np.array(x1), np.array(x2)

def ins_gen_train_batch_yxx(_data, batch_size):
  part_one, part_two = int(batch_size / 4 * 3), int(batch_size / 4)
  psample = random.sample(_data, part_one)
  nsample = random.sample(_data, part_two)
  y = [1.0] * part_one + [0.0] * part_two
  x1 = [s1 for s1, s2 in psample] + [s1 for s1, s2 in psample[:part_two]]
  x2 = [s2 for s1, s2 in psample] + [s2 for s1, s2 in nsample]
  return np.array(y), np.array(x1), np.array(x2)

def gen_test_batch_qpn(_data, start, end):
  sample = _data[start:end]
  for i in range(len(sample), end - start):
    sample.append(sample[-1])
  f = [int(i) for i,_,_,_ in sample]
  g = [int(i) for _,i,_,_ in sample]
  q1 = [i for _,_,i,_ in sample]
  q2 = [i for _,_,_,i in sample]
  return f, g, np.array(q1), np.array(q2)

def gen_test_batch_yxx(_data, start, end):
  if config.dataset == config.dataset_ins:
    return ins_gen_test_batch_yxx(_data, start, end)
  if config.dataset == config.dataset_qur:
    return qur_gen_test_batch_yxx(_data, start, end)
  print('bad gen_test_batch_yxx')
  exit(1)

def qur_gen_test_batch_yxx(_data, start, end):
  sample = _data[start:end]
  y = [i for i,_,_ in sample]
  x1 = [i for _,i,_ in sample]
  x2 = [i for _,_,i in sample]
  return y, y, np.array(x1), np.array(x2)

def ins_gen_test_batch_yxx(_data, start, end):
  sample = _data[start:end]
  for i in range(len(sample), end - start):
    sample.append(sample[-1])
  f = [int(i) for i,_,_,_ in sample]
  g = [int(i) for _,i,_,_ in sample]
  q1 = [i for _,_,i,_ in sample]
  q2 = [i for _,_,_,i in sample]
  return f, g, np.array(q1), np.array(q2)

def _eval(y, g, yp):
  if config.dataset == config.dataset_ins:
    eval_auc(y, g, yp)
    eval_top1_prec(y, g, yp)
  if config.dataset == config.dataset_qur:
    eval_auc(y, g, yp)
    eval_best_prec(y, g, yp)

def eval_best_prec(y, g, yp):
  best_p, best_s = 0.0, 0.0
  for i in range(50, 100, 1):
    i = float(i) / 100
    positive = 0
    for _y, _yp in zip(y, yp):
      p = 1 if _yp >= i else 0
      if p == _y: positive += 1
    prec = positive / len(yp)
    if prec > best_p:
      best_p = prec
      best_s = i
  print('best_prec: ' + str(best_p) + ' best_threshold:' + str(best_s))
  return best_p, best_s

def eval_auc(y, g, yp):
  auc = metrics.roc_auc_score(y, yp)
  print('auc: ' + str(auc))
  return auc

def eval_top1_prec(y, g, yp):
  _list = [(_y, _g, _yp) for _y, _g, _yp in zip(y, g, yp)]
  _dict = {}
  for _y, _g, _yp in _list:
    if not _g in _dict: _dict[_g] = []
    _dict[_g].append((_y, _g, _yp))
  positive, gc = 0 , 0
  for _, group in _dict.items():
    group = sorted(group, key=itemgetter(2), reverse=True)
    gc += 1
    if group[0][0] == 1: 
      positive += 1
  prec = positive / gc
  print('top1 precision ' + str(positive) + '/' + str(gc) + ': '+ str(positive / gc))
  return prec

