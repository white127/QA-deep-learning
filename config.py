import os

dataset_ins = 'insurance-qa'
dataset_qur = 'quora-qa'

##################################################################
# ajust to your runnning environment
# which data do you want
dataset = dataset_qur
# word2vec command path
w2v_command = '/export/jw/word2vec/word2vec'
##################################################################

home = ''
if dataset == dataset_ins:
  home = os.path.expanduser('/export/jw/insuranceQA')
elif dataset == dataset_qur:
  home = os.path.expanduser('/export/jw/quora')

#Insurance-QA original data directory
qa_version = 'V1'
vocab_file = os.path.join(home, qa_version, 'vocabulary')
answers_file = os.path.join(home, qa_version, 'answers.label.token_idx')
question_train_file = os.path.join(home, qa_version, 'question.train.token_idx.label')
question_test1_file = os.path.join(home, qa_version, 'question.test1.label.token_idx.pool')
question_test2_file = os.path.join(home, qa_version, 'question.test2.label.token_idx.pool')
question_dev_file = os.path.join(home, qa_version, 'question.dev.label.token_idx.pool')
#quora original data directory
qr_file = os.path.join(home, 'quora_duplicate_questions.tsv')
qr_train_ratio = 0.8
#processed files
train_file = os.path.join(home, 'data', 'train.prepro')
test1_file = os.path.join(home, 'data', 'test1.prepro')
test2_file = os.path.join(home, 'data', 'test2.prepro')
w2v_train_file = os.path.join(home, 'data', 'w2v.train')
w2v_bin_file = os.path.join(home, 'data', 'w2v.bin')
predict1_file = os.path.join(home, 'data', 'predict1')
  
