Insurance-QA deeplearning model
======
This is a repo for Q&A Mathing, includes some deep learning models, such as CNN、RNN.<br>
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
   
    python3 gen.py<br>
    
Run CNN model<br>

    cd ./cnn/tensorflow && python3 insqa_train.py
    
It will take few hours(thousands of epoches) to train this model on a single GPU.<br>
    
## Others
1. You can get Insurance-QA data from here https://github.com/shuzi/insuranceQA<br>

## Links
1. CNN and RNN textual classification  https://github.com/white127/TextClassification_CNN_RNN<br>


