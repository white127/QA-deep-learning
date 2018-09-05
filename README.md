Insurance-QA deeplearning model
======
This is a repo for Q&A Mathing, includes some deep learning models, such as CNN、RNN.<br>
1. CNN. Basic CNN model from 《Applying Deep Learning To Answer Selection: A Study And An Open Task》<br>
2. RNN. RNN seems the best model on Insurance-QA dataset.<br>
3. SWEM. SWEM is the fastest, and has good effect on other datasets, such as TrecQA ..., but is seems not so good on Insurance-QA dataset<br>


It's hard to say which model is the best in other datasets, you have to choose the most suitable model for your datasets.<br>
More models are on the way, pay attention to the updates.<br>

## Requirements
1. tensorflow 1.4.0<br>
2. python3.5<br>

## Performance
margin loss version<br>

Model/Score | top1_precision
------------ | -------------
CNN | 62%
LSTM+CNN | 68%

logloss version<br>

## Running
Change configuration to your own environment, just like data pathes<br>
    
    vim config.py

Data processing<br>
   
    python3 gen.py
    
Run CNN model<br>

    cd ./cnn/tensorflow && python3 insqa_train.py
    
It will take few hours(thousands of epoches) to train this model on a single GPU.<br>
    
## Others
1. You can get Insurance-QA data from here https://github.com/shuzi/insuranceQA<br>

## Links
1. CNN and RNN textual classification repo  https://github.com/white127/TextClassification_CNN_RNN<br>
2. 《Applying Deep Learning To Answer Selection: A Study And An Open Task》<br>

