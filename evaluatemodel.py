import json
#from func_utility import find_answer_in_tweet ,create_batche
#from twitter_qa import TwitterQa
#from  model.func_utility import *
from model.func_utility import *
from model.twitter_qa2 import *
from model.twitter_batch import *
from model.compute_f1_callback import *
from model.tweetqa_eval import *

import os

#os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
#TRAIN_PATH = "./TweetQA_data/tweet_squad_train_preprocess.json"
#TEST_PATH = "./TweetQA_data/test.json"
#TEST_PATH = "./TweetQA_data/squad_train.json"
TEST_PATH = "./TweetQA_data/tweet_train.json"
#TRAIN_PATH= TEST_PATH
#TRAIN = True
with open(TEST_PATH) as f:
    data = json.load(f)
#data =find_answer_in_tweet2(data=data)
data =find_answer_in_tweet(data=data)
#data =find_answer_in_tweet2(data=data)
testing_data_length = int(len(data) * .10)
#testing_data_length = int(len(data) * .20)
#testing_data_length = int(len(data) * .5)
#testing_data_length = int(len(data))
#batch = create_batche(data[:data_used])
batch = create_batche(data[:testing_data_length])
twitterqa = TwitterQa(200,3e-5,load_model=True)
#twitterqa.load_weights()
#twitterqa.save_model()
#twitterqa.load_model()
predictions = twitterqa.predict(batch)
#for a in range(testing_data_length):
    #data[-testing_data_length + a]["Answer"] = [data[-testing_data_length + a]["Answer"][0].split()]
    #data[-testing_data_length + a]["Answer"] = [data[-testing_data_length + a]["Answer"]]
true_data = []
prediction_data = []
for a in range(batch.batch_length):
    tweet = {}
    tweet_p = {}
    tweet["Tweet"]= batch.tweets[a]
    tweet["Question"]= batch.questions[a]
    tweet["Answer"]= [batch.answers[a]]
    tweet_p["Tweet"]= batch.tweets[a]
    tweet_p["Question"]= batch.questions[a]
    if predictions:
        tweet_p["Answer"]= predictions[a][2]
    else:
        tweet_p["Answer"]= ""

    tweet["qid"]= batch.qid[a]
    tweet_p["qid"]= batch.qid[a]
    true_data.append(tweet)
    prediction_data.append(tweet_p)
with open("gold_file.json","w") as f:
    #json.dump(data[-testing_data_length:],f)
    json.dump(true_data,f)
with open("prediction_file.json","w") as f:
    #json.dump(data[-testing_data_length:],f)
    json.dump(prediction_data,f)

evaluate("gold_file.json","prediction_file.json","test")
    



