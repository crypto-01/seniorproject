import json
#from func_utility import find_answer_in_tweet ,create_batche
#from twitter_qa import TwitterQa
from  model.func_utility import *
from model.twitter_qa import *
from model.twitter_batch import *
from model.compute_f1_callback import *
from model.tweetqa_eval import *

import os

#os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
TRAIN_PATH = "./TweetQA_data/train.json"
SQUAD_TRAIN_PATH = "./TweetQA_data/squad1.1/train-v1.1.json"
COMBINED_TWEET_SQUAD="./TweetQA_data/tweet_squad_train.json"
TWEET_TRAIN = "./TweetQA_data/tweet_train.json"
SQUAD_TRAIN = "./TweetQA_data/squad_train.json"
DEMO = "./TweetQA_data/demo.json"
#TEST_PATH = "./TweetQA_data/test.json"
#TRAIN = False

with open(TRAIN_PATH) as f:
    data = json.load(f)
tweet_data =find_answer_in_tweet(data=data)

with open(SQUAD_TRAIN_PATH) as f:
    data = json.load(f)
    #print(len(data))
    print(data.keys())
    squad_data = process_squid_data(data["data"])
print(len(squad_data))
print(len(tweet_data))
print(len(tweet_data) + len(squad_data))
all_data = squad_data + tweet_data
with open(COMBINED_TWEET_SQUAD,"w") as f:
    json.dump(all_data,f)
with open(TWEET_TRAIN,"w") as f:
    json.dump(tweet_data,f)
with open(DEMO,"w") as f:
    json.dump(tweet_data[:1000],f)
with open(SQUAD_TRAIN,"w") as f:
    json.dump(squad_data,f)
"""
for a in range(50):
    answer_start = tweet_data[a]["answer_start"]
    answer_length = tweet_data[a]["answer_length"]
    print(tweet_data[a]["Answer"])
    print([tweet_data[a]["Tweet"][answer_start: answer_start + answer_length]])
    print(tweet_data[a]["groups"])
    print(tweet_data[a]["group_text"])
    print()
    """




