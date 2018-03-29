#!/usr/bin/env python3.4

import os

wordmap = {}
for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/vocabulary'):
    items = line.strip().split('\t')
    wordmap[items[0]] = items[1]

def word_trans(string):
    newstr = string.replace('_', '-')
    return newstr

def align(string, length):
    global wordmap
    out = ''
    words = string.strip().split(' ')
    idx = int(0)
    for i in range(0, length):
        if i < len(words):
            #out += (word_trans(words[i]) + '_')
            out += (wordmap[words[i]] + '_')
        else:
            out += '<a>_'
    return out

def align_space(string):
    out = ''
    words = string.strip().split(' ')
    for w in words:
        #out += (word_trans(w) + ' ')
        out += (wordmap[w] + ' ')
    return out

def load_answers():
    #answer id begins from 1, not 0
    ansList = []
    ansList.append('<null>')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/answers.label.token_idx'):
        items = line.strip().split('\t')
        ansList.append(items[1])
    return ansList

def train():
    ansList = load_answers()
    w2v = '/export/jw/cnn/insuranceQA/w2v.train'
    of = open('/export/jw/cnn/insuranceQA/train', 'w')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.train.token_idx.label'):
        items = line.strip().split('\t')
        ansidList = items[1].split(' ')
        for ansid in ansidList:
            of.write('1 qid:0 ')
            of.write(align(items[0], 200) + ' ')
            of.write(align(ansList[int(ansid)], 200) + '\n')
    of.close()

def w2v():
    ansList = load_answers()
    of = open('/export/jw/cnn/insuranceQA/w2v.train', 'w')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.train.token_idx.label'):
        items = line.strip().split('\t')
        of.write(align_space(items[0]) + '\n')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/answers.label.token_idx'):
        items = line.strip().split('\t')
        of.write(align_space(items[1]) + '\n')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.dev.label.token_idx.pool'):
        items = line.strip().split('\t')
        of.write(align_space(items[1]) + '\n')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.test1.label.token_idx.pool'):
        items = line.strip().split('\t')
        of.write(align_space(items[1]) + '\n')
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.test2.label.token_idx.pool'):
        items = line.strip().split('\t')
        of.write(align_space(items[1]) + '\n')
    of.close()

def test1():
    ansList = load_answers()
    of = open('/export/jw/cnn/insuranceQA/test1', 'w')
    qid = int(0)
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.test1.label.token_idx.pool'):
        items = line.strip().split('\t')
        truthmap = {}
        for tid in items[0].split(' '):
            truthmap[tid] = '#'
        quest = align(items[1], 200)
        ansidList = items[2].split(' ')
        for ansid in ansidList:
            if ansid in truthmap:
                of.write('1 ')
            else:
                of.write('0 ')
            of.write('qid:' + str(qid) + ' ')
            of.write(quest + ' ')
            of.write(align(ansList[int(ansid)], 200) + '\n')
        qid += 1
    of.close()

def test2():
    ansList = load_answers()
    of = open('/export/jw/cnn/insuranceQA/test2', 'w')
    qid = int(0)
    for line in open('/export/jw/cnn/insuranceQA/insuranceQA-master/question.test2.label.token_idx.pool'):
        items = line.strip().split('\t')
        truthmap = {}
        for tid in items[0].split(' '):
            truthmap[tid] = '#'
        quest = align(items[1], 200)
        ansidList = items[2].split(' ')
        for ansid in ansidList:
            if ansid in truthmap:
                of.write('1 ')
            else:
                of.write('0 ')
            of.write('qid:' + str(qid) + ' ')
            of.write(quest + ' ')
            of.write(align(ansList[int(ansid)], 200) + '\n')
        qid += 1
    of.close()

if __name__  == '__main__':
    w2v()
    train()
    test1()
    test2()

