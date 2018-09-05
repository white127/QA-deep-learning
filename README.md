Insurance-QA deeplearning model
======

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

Data processed<br>
   
    python3 gen.py<br>
    
Run CNN model<br>

    cd ./cnn/tensorflow && python3 insqa_train.py
    
## Others
1. You can get Insurance-QA data from here https://github.com/shuzi/insuranceQA

## Links
1. CNN and RNN textual classification  https://github.com/white127/TextClassification_CNN_RNN
【insuranceQA-cnn-lstm README】

