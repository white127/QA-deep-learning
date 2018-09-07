import config, os, random

#####################################################################
# function: load vocab
# return: dict[word] = [word_id]
#####################################################################
def load_vocab():
  voc = {}
  for line in open(config.vocab_file):
    word, _id = line.strip().split('\t')
    voc[word] = _id
  return voc

#####################################################################
# function: load answers, restore idx to real word
# return : [answer_1, answer_2, ..., answer_n]
#####################################################################
def ins_load_answers():
  _list, voc = ['<None>'], load_vocab()
  for line in open(config.answers_file):
    _, sent = line.strip().split('\t')
    _list.append('_'.join([voc[wid] for wid in sent.split(' ')]))
  return _list

#####################################################################
# function: preprea word2vec binary file
# return : 
#####################################################################
def ins_w2v():
  print('preparing word2vec ......')
  _data, voc = [], load_vocab()
  for line in open(config.question_train_file):
    items = line.strip().split('\t')
    _data.append(' '.join([voc[_id] for _id in items[0].split(' ')]))
  for _file in [config.answers_file, config.question_dev_file, \
          config.question_test1_file, config.question_test2_file]:
    for line in open(_file):
      items = line.strip().split('\t')
      _data.append(' '.join([voc[_id] for _id in items[1].split(' ')]))
  of = open(config.w2v_train_file, 'w')
  for s in _data: of.write(s + '\n')
  of.close()
  os.system('time ' + config.w2c_command + ' -train ' + config.w2v_train_file + ' -output ' + config.w2v_bin_file + ' -cbow 0 -size 100 -window 5 -negative 20 -sample 1e-3 -threads 12 -binary 0 -min-count 1')

#####################################################################
# function: preprea train file
# file format: flag question answer
#####################################################################
def ins_train():
  print('preparing train ......')
  answers, voc, _data = ins_load_answers(), load_vocab(), []
  for line in open(config.question_train_file):
    qsent, ids = line.strip().split('\t')
    qsent = '_'.join([voc[wid] for wid in qsent.split(' ')])
    for _id in ids.split(' '):
      _data.append(' '.join(['1', qsent, answers[int(_id)]]))
  of = open(config.train_file, 'w')
  for _s in _data: of.write(_s + '\n')
  of.close()

#####################################################################
# function: preprea test file
# file format: flag group_id question answer
#####################################################################
def ins_test():
  print('preparing test ......')
  answers, voc = ins_load_answers(), load_vocab()
  for _in, _out in ([(config.question_test2_file, config.test2_file), \
          (config.question_test1_file, config.test1_file)]):
    _data, group = [], int(0)
    for line in open(_in):
      pids, qsent, pnids = line.strip().split('\t')
      positive = {_id:'#' for _id in pids.split(' ')}
      qsent = '_'.join([voc[wid] for wid in qsent.split(' ')])
      for _id in pnids.split(' '):
        flag = '1' if _id in positive else '0'
        _data.append(' '.join([flag, str(group), qsent, answers[int(_id)]]))
      group += 1
    of = open(_out, 'w')
    for s in _data: of.write(s + '\n')
    of.close()

def ins_qa():
  ins_w2v()
  ins_train()
  ins_test()

def qur_prepare():
  #pretrain word2vec
  _list = []
  for line in open(config.qr_file):
    items = line.strip().split('\t')
    if len(items) != 6:
      continue
    _list.append(items)
  _list = _list[1:]
  random.shuffle(_list)
  _list = [(f, q1, q2) for _,_,_,q1,q2,f in _list]
  of = open(config.w2v_train_file, 'w')
  for f, q1, q2 in _list:
    of.write(q1 + '\n')
    of.write(q2 + '\n')
  of.close()
  os.system('time ' + config.w2v_command + ' -train ' + config.w2v_train_file + ' -output ' + config.w2v_bin_file + ' -cbow 0 -size 100 -window 5 -negative 20 -sample 1e-3 -threads 12 -binary 0 -min-count 1')
  #train file
  _newlist = []
  for f, q1, q2 in _list:
    if len(q1) <= 1 or len(q2) <= 1: continue
    q1 = '_'.join(q1.split(' '))
    q2 = '_'.join(q2.split(' '))
    _newlist.append((f, q1, q2))
  _list = _newlist
  of = open(config.train_file, 'w')
  for f, q1, q2 in _list[:int(len(_list) * 0.8)]:
    of.write(' '.join([f, q1, q2]) + '\n')
  of.close()

  #test file
  of = open(config.test1_file, 'w')
  for f, q1, q2 in _list[int(len(_list) * 0.8):]:
    of.write(' '.join([f, q1, q2]) + '\n')
  of.close()

def qur_qa():
  qur_prepare()

if __name__ == '__main__':
  if config.dataset == config.dataset_ins:
    ins_qa()
  elif config.dataset == config.dataset_qur:
    qur_qa()
