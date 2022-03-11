import string
import re
#from twitter_batch import TwitterBatch
from model.twitter_batch import *


def normalize_text(text):
    text = text.lower()
    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text

def find_answer_in_tweet(data):
    exact_match = 0
    no_match =[]
    no_match2=[]
    no_match3=[]
    success = []
    success2 = []
    success3 = []
    #first pass
    if "Answer" not in data[0].keys():
        return data
    for idx in range(len(data)):
        tweet = data[idx]["Tweet"].lower()
        answer = data[idx]["Answer"][0].lower().strip(".")
        if( answer in tweet):
            exact_match +=1
            data[idx]["answer_start"] = tweet.index(answer)
            data[idx]["answer_length"] = len(answer)
            success.append(data[idx])
        else:
            no_match.append(data[idx])

    #second pass
    for a in no_match:
        tweet = a["Tweet"].lower()
        answer = a["Answer"][0].lower().strip(".")
        answer =  answer.replace(" ","")
        if answer in tweet:
            a["answer_start"] = tweet.index(answer)
            a["answer_length"] = len(answer)
            success2.append(a)
            exact_match +=1
        else:
            no_match2.append(a)

    #third  pass
    for a in no_match2:
        tweet = a["Tweet"].lower()
        answer =  a["Answer"][0].lower()
        answer = normalize_text(answer)
        if answer in tweet:
            a["answer_start"] = tweet.index(answer)
            a["answer_length"] = len(answer)
            success3.append(a)
            exact_match +=1
        else:
            #print(tweet)
            #print("Answer: ",answer)
            #print()
            no_match3.append(a)
    return success + success2 + success3

def create_batche(data):
    tweets = [item.get("Tweet").lower() for item in data]
    questions = [item.get("Question").lower() for item in data]
    if "Answer" in data[0].keys():
        answers = [item.get("Answer")[0].lower() for item in data]
        answers_start = [item.get("answer_start") for item in data]
        answers_length = [item.get("answer_length") for item in data]
    else:
        answers = []
        answers_start = []
        answers_length = []

    batch = TwitterBatch(questions,tweets,answers,answers_start,answers_length)
    batch.preprocess()
    return batch
