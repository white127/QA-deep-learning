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
  _data = []
  for line in open(config.train_file):
    f, q1, q2 = line.strip().split(' ')
    q1, q2 = encode_sent(q1, vocab, max_len), encode_sent(q2, vocab, max_len)
    _data.append((q1, q2))
  return _data

def load_test_data(vocab, max_len):
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
  psample = random.sample(_data, int(batch_size / 2))
  nsample = random.sample(_data, int(batch_size / 2))
  y = [1.0] * int(batch_size / 2) + [0.0] * int(batch_size / 2)
  x1 = [s1 for s1, s2 in psample] + [s1 for s1, s2 in psample]
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
  sample = _data[start:end]
  for i in range(len(sample), end - start):
    sample.append(sample[-1])
  f = [int(i) for i,_,_,_ in sample]
  g = [int(i) for _,i,_,_ in sample]
  q1 = [i for _,_,i,_ in sample]
  q2 = [i for _,_,_,i in sample]
  return f, g, np.array(q1), np.array(q2)

def eval_auc(y, g, yp):
  score = metrics.roc_auc_score(y, yp)
  print('auc: ' + str(score))
  return score

"""
def eval_top1_prec(y, g, yp):
  _list = [(_y, _g, _yp) for _y, _g, _yp in zip(y, g, yp)]
  group = groupby(_list, itemgetter(1))  
  positive, gc = 0 , 0
  for key, _g in group:
    rank = list(_g)
    rank = sorted(rank, key=itemgetter(2), reverse=True)
    if rank[0][0] == 1: positive += 1
    gc += 1
  print('top1 precision ' + str(positive) + '/' + str(gc) + ': '+ str(positive / gc))
"""

def _eval_top1_prec(y, g, yp):
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
  print('top1 precision ' + str(positive) + '/' + str(gc) + ': '+ str(positive / gc))

