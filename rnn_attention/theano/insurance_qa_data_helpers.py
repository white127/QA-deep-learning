import numpy as np
import random
from operator import itemgetter

precision = '/export/jw/cnn/insuranceQA/acc.lstm'

empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))

def load_word_embeddings(vocab, dim):
    embeddings = [] #brute initialization
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, dim):
            vec.append(0.01)
        embeddings.append(vec)
    return np.array(embeddings, dtype='float32')

def build_vocab():
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open('/export/jw/cnn/insuranceQA/train'):
        items = line.strip().split(' ')
        for i in range(2, 3):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open('/export/jw/cnn/insuranceQA/test1'):
        items = line.strip().split(' ')
        for i in range(2, 3):
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab

def read_alist():
    alist = []
    for line in open('/export/jw/cnn/insuranceQA/train'):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist

def load_vectors():
    vectors = {}
    for line in open('/export/jw/cnn/insuranceQA/vectors.nobin'):
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

def read_vector(vectors, word):
    global empty_vector
    if word in vectors:
        return vectors[word]
    else:
        return empty_vector
        #return vectors['</s>']

def load_train_list():
    train_list = []
    for line in open('/export/jw/cnn/insuranceQA/train'):
        items = line.strip().split(' ')
        if items[0] == '1':
            train_list.append(line.strip().split(' '))
    return train_list

def load_test_list():
    test_list = []
    for line in open('/export/jw/cnn/insuranceQA/test1'):
       test_list.append(line.strip().split(' '))
    return test_list

def load_train_and_vectors():
    trainList = []
    for line in open('/export/jw/cnn/insuranceQA/train'):
        trainList.append(line.strip())
    vectors = load_vectors()
    return trainList, vectors

def read_raw():
    raw = []
    for line in open('/export/jw/cnn/insuranceQA/train'):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw

def encode_sent(vocab, string, size):
    x, m = [], []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab: x.append(vocab[words[i]])
        else: x.append(vocab['UNKNOWN'])
        if words[i] == '<a>': m.append(1)
        else: m.append(1)
    return x, m

def load_val_data(test_list, vocab, index, batch_size, max_len):
    x1, x2, x3, m1, m2, m3 = [], [], [], [], [], []
    for i in range(0, batch_size):
        t_i = index + i
        if t_i >= len(test_list):
            t_i = len(test_list) - 1
        items = test_list[t_i]
        x, m = encode_sent(vocab, items[2], max_len)
        x1.append(x)
        m1.append(m)
        x, m = encode_sent(vocab, items[3], max_len)
        x2.append(x)
        m2.append(m)
        x, m = encode_sent(vocab, items[3], max_len)
        x3.append(x)
        m3.append(m)
    return np.transpose(np.array(x1, dtype='float32')), np.transpose(np.array(x2, dtype='float32')), np.transpose(np.array(x3, dtype='float32')), np.transpose(np.array(m1, dtype='float32')) , np.transpose(np.array(m2, dtype='float32')), np.transpose(np.array(m3, dtype='float32'))

def load_train_data(trainList, vocab, batch_size, max_len):
    train_1, train_2, train_3 = [], [], []
    mask_1, mask_2, mask_3 = [], [], []
    counter = 0
    while True:
        pos = trainList[random.randint(0, len(trainList)-1)]
        neg = trainList[random.randint(0, len(trainList)-1)]
        if pos[2].startswith('<a>') or pos[3].startswith('<a>') or neg[3].startswith('<a>'):
            #print 'empty string ......'
            continue
        x, m = encode_sent(vocab, pos[2], max_len)
        train_1.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, pos[3], max_len)
        train_2.append(x)
        mask_2.append(m)
        x, m = encode_sent(vocab, neg[3], max_len)
        train_3.append(x)
        mask_3.append(m)
        counter += 1
        if counter >= batch_size:
            break
    return np.transpose(np.array(train_1, dtype='float32')), np.transpose(np.array(train_2, dtype='float32')), np.transpose(np.array(train_3, dtype='float32')), np.transpose(np.array(mask_1, dtype='float32')) , np.transpose(np.array(mask_2, dtype='float32')), np.transpose(np.array(mask_3, dtype='float32'))

def evaluation(score_list, test_list):
    global precision
    sessdict, index = {}, int(0)
    for items in test_list:
        qid = items[1].split(':')[1]
        if not qid in sessdict:
            sessdict[qid] = []
        sessdict[qid].append((score_list[index], items[0]))
        index += 1
        if index >= len(test_list):
            break
    lev1, lev0 = float(0), float(0)
    of = open(precision, 'a')
    for k, v in sessdict.items():
        v.sort(key=itemgetter(0), reverse=True)
        score, flag = v[0]
        if flag == '1': lev1 += 1
        if flag == '0': lev0 += 1
    of.write('lev1:' + str(lev1) + '\n')
    of.write('lev0:' + str(lev0) + '\n')
    print('lev1 ' + str(lev1))
    print('lev0 ' + str(lev0))
    print('precision:' + str(lev1 / (lev0 + lev1)))
    of.close()
