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
TRAIN_PATH_TWEET_SQUAD = "./TweetQA_data/tweet_squad_train_preprocess.json"
TRAIN_PATH_SQUAD = "./TweetQA_data/squad_train.json"
TRAIN_PATH_TWEET = "./TweetQA_data/tweet_train.json"
#TRAIN_PATH = TRAIN_PATH_SQUAD

#TEST_PATH = "./TweetQA_data/test.json"
LOAD_WEIGHTS = False
#LOAD_WEIGHTS = True
#TRAIN = False

#EPOCHS =30
#EPOCHS = 100
#EPOCHS =10
EPOCHS =5
BATCH_SIZE = 8
with open(TRAIN_PATH) as f:
    data = json.load(f)
#with open(TRAIN_PATH_SQUAD) as f:
#    data_squad = json.load(f)
#with open(TRAIN_PATH_TWEET) as f:
#    data_tweet = json.load(f)

#data =find_answer_in_tweet2(data=data)
data =find_answer_in_tweet(data=data)
testing_data_length = int(len(data) * .10)
#data = data + data_squad[:50000]
#data = data + data_squad
#randomize = np.arange(len(data))
#data = data[randomize.tolist()]
#data = data_tweet
#testing_data_length = int(len(data) * .10)
validation_data = data[:testing_data_length]
data = data[testing_data_length:]
data2 = []
randomize = np.arange(len(data))
np.random.shuffle(randomize)
for a in randomize:
    data2.append(data[a])
data = data2
#data = data[:1000]
#testing_data_length = int(len(data) * .06)
#data = data[-15000:] + data[:-15000]
#data =  data[:-15000]
#testing_data_length = int(len(data_tweet) * .20)
#data = data_squad[:30000] + data_tweet[:len(data_tweet) - testing_data_length]
#data = data_squad + data_tweet[:len(data_tweet) - testing_data_length]
#data = data_squad + data_tweet[testing_data_length:]
#data = data_squad
train_batch = create_batche(data[:len(data)-testing_data_length])
#train_batch = create_batche(data)
#test_batch = create_batche(data[-testing_data_length:])
test_batch = create_batche(validation_data)

#test_batch = create_batche(data_tweet[:testing_data_length])
print(train_batch.batch_length)
#weights_start = get_class_weights(train_batch.start_tokens_idx.flatten(),200)
#weights_end = get_class_weights(train_batch.end_tokens_idx.flatten(),200)
#twitterqa = TwitterQa(200,2e-7)
#twitterqa = TwitterQa(200,3.5e-5,(weights_start,weights_end))
#twitterqa = TwitterQa(200,3.5e-4)
#twitterqa = TwitterQa(200,0.5e-6)
#twitterqa = TwitterQa(200,5.0e-6) best
#twitterqa = TwitterQa(200,1.0e-5)
if LOAD_WEIGHTS:
    twitterqa = TwitterQa(200,5.0e-7,load_model=True)
else:
    twitterqa = TwitterQa(200,5.0e-7,load_model=False)
twitterqa.train(train_batch,test_batch,EPOCHS,BATCH_SIZE)
twitterqa.save_model()
