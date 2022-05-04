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
TRAIN_PATH = "./TweetQA_data/train.json"
#TRAIN_PATH = "./TweetQA_data/tweet_squad_train_preprocess.json"
#TEST_PATH = "./TweetQA_data/test.json"
TEST_PATH = TRAIN_PATH
#TEST_PATH = "./TweetQA_data/squad_train.json"
TEST_PATH = "./TweetQA_data/tweet_train.json"
#TRAIN_PATH= TEST_PATH
#TRAIN = True
TRAIN = False

with open(TEST_PATH) as f:
    data = json.load(f)
#data =find_answer_in_tweet2(data=data)
data =find_answer_in_tweet(data=data)
#data =find_answer_in_tweet2(data=data)
#testing_data_length = int(len(data) * 08)
#testing_data_length = int(len(data) * .03)
testing_data_length = int(len(data) * .1)
#testing_data_length = int(len(data) * .20)
#testing_data_length = int(len(data) * .5)
#testing_data_length = 400
#testing_data_length = int(len(data))
#batch = create_batche(data[:data_used])
batch = create_batche(data[:testing_data_length])
#batch = create_batche(data[:30])
twitterqa = TwitterQa(200,3e-5,load_model=True)
#twitterqa.load_weights()
predictions = twitterqa.predict(batch)
true_data = []
prediction_data = []
tp_total =0
fp_total = 0
fn_total = 0
all_f1_scores = []
for a in range(batch.batch_length):
    #precision,recall,f1= calculate_confusion_and_f1("test","false")
    results = calculate_confusion_and_f1_score(batch.answers[a],predictions[a][2])
    all_f1_scores.append(results[0][2])
    tp_total += results[1][0]
    fp_total += results[1][1]
    fn_total  += results[1][2]
    #p_r_f1,confusion= calculate_confusion_and_f1(batch.answers[a],predictions[a][2])
print("TP num_of_same_tokens                       :",tp_total)
print("fp num of tokens not in correct answer      :",fp_total)
print("fn  num of correct tokens not in predictions:",fn_total)
print("F1 score                                    :",sum(all_f1_scores)/len(all_f1_scores))

"""
for a in range(batch.batch_length):
    tweet = {}
    tweet_p = {}
    tweet["Tweet"]= batch.tweets[a]
    tweet["Question"]= batch.questions[a]
    tweet["Answer"]= [batch.answers[a]]
    tweet_p["Tweet"]= batch.tweets[a]
    tweet_p["Question"]= batch.questions[a]
    tweet_p["Answer"]= predictions[a][2]
    tweet["qid"]= batch.qid[a]
    tweet_p["qid"]= batch.qid[a]
    true_data.append(tweet)
    prediction_data.append(tweet_p)
with open("gold_file.json","w") as f:
    #json.dump(data[-testing_data_length:],f)
    json.dump(true_data,f)
#for a in range(len(predictions)):
#    data[-testing_data_length + a]["Answer"] = [predictions[a][2]]
with open("prediction_file.json","w") as f:
    #json.dump(data[-testing_data_length:],f)
    json.dump(prediction_data,f)

evaluate("gold_file.json","prediction_file.json","test")
print(testing_data_length,batch.batch_length,len(batch.input_ids),len(batch.mask_ids),len(batch.segment_ids),len(prediction_data))
    
"""
    

#twitterqa.train(train_batch,test_batch,EPOCHS,batch_size)
#twitterqa.save_weights()
    
