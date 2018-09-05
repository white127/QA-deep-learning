import os

home = os.path.expanduser('/export/jw/insuranceQA')
qa_version = 'V1'
vocab_file = os.path.join(home, qa_version, 'vocabulary')
answers_file = os.path.join(home, qa_version, 'answers.label.token_idx')
question_train_file = os.path.join(home, qa_version, 'question.train.token_idx.label')
question_test1_file = os.path.join(home, qa_version, 'question.test1.label.token_idx.pool')
question_test2_file = os.path.join(home, qa_version, 'question.test2.label.token_idx.pool')
question_dev_file = os.path.join(home, qa_version, 'question.dev.label.token_idx.pool')
train_file = os.path.join(home, 'data', 'train.prepro')
test1_file = os.path.join(home, 'data', 'test1.prepro')
test2_file = os.path.join(home, 'data', 'test2.prepro')
w2v_train_file = os.path.join(home, 'data', 'w2v.train')
w2v_bin_file = os.path.join(home, 'data', 'w2v.bin')
predict1_file = os.path.join(home, 'data', 'predict1')
