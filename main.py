import json
#from func_utility import find_answer_in_tweet ,create_batche
#from twitter_qa import TwitterQa
#from  model.func_utility import *
from model.func_utility import *
from model.twitter_qa import *
from model.twitter_batch import *
from model.compute_f1_callback import *
from model.tweetqa_eval import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
TRAIN_PATH = "./TweetQA_data/train.json"
#TEST_PATH = "./TweetQA_data/test.json"
TEST_PATH = TRAIN_PATH
TRAIN = False

if TRAIN:
    #EPOCHS =10
    #BATCH_SIZE = 4
    EPOCHS = 1
    BATCH_SIZE = 1
    with open(TRAIN_PATH) as f:
        data = json.load(f)
    data =find_answer_in_tweet(data=data)
    testing_data_length = int(len(data) * .06)
    train_batch = create_batche(data[:testing_data_length])
    test_batch = create_batche(data[testing_data_length:])
    twitterqa = TwitterQa(200,5e-5)
    
    twitterqa.load_weights()
    
    twitterqa.train(train_batch,test_batch,EPOCHS,BATCH_SIZE)
else:
    with open(TEST_PATH) as f:
        data = json.load(f)
    data =find_answer_in_tweet(data=data)
    testing_data_length = int(len(data) * .06)
    #testing_data_length = 400
    #testing_data_length = int(len(data))
    #batch = create_batche(data[:data_used])
    batch = create_batche(data[-testing_data_length:])
    twitterqa = TwitterQa(200,5e-5)
    
    twitterqa.load_weights()
    
    predictions = twitterqa.predict(batch)
    #print()
    #print("Tweet: ",predictions[9][0])
    #print("Question: ",predictions[9][1])
    #print("Answer: ",predictions[9][2])
    #print()
    
    with open("gold_file.json","w") as f:
        json.dump(data[-testing_data_length:],f)
    for a in range(testing_data_length):
        data[-testing_data_length + a]["Answer"] = [predictions[a][2]]
    with open("prediction_file.json","w") as f:
        json.dump(data[-testing_data_length:],f)

    evaluate("gold_file.json","prediction_file.json","test")
        
    
        

#twitterqa.train(train_batch,test_batch,EPOCHS,batch_size)
#twitterqa.save_weights()
    
