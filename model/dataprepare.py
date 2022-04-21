import string
import re
#from twitter_batch import TwitterBatch
from model.twitter_batch import *
import re


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

def splits_with_index(text, delim):
    final_splits =[]
    text_split = text.split(delim)
    index = 0
    for word in text_split:
        final_splits.append((word,index))
        index += len(word) + len(delim)
    return final_splits

def find_answer_in_tweet(data):
    success = []
    success2 = []
    fails = []
    fails2 = []
    if "Answer" not in data[0].keys():
        return data
    for idx in range(len(data)):
        tweet = data[idx]["Tweet"].lower()
        answer = data[idx]["Answer"][0].lower().strip(".")
        answer = answer.split()
        rg_expression = r"(" + "|".join(answer) + ")"
        for a in range(len(answer)):
            #rg_expression = rg_expression + r"(?:\W+\w+){0,6}?\W+(" + ("|".join(answer)) + r")?"
            #rg_expression = rg_expression + r"(?:\W+\w+){0,6}?\W{0,3}(" + ("|".join(answer)) + r")?"
            rg_expression = rg_expression + r"(?:\W+\w+){0,6}?\W{0,3}(" + ("|".join(answer)) + r")?"
        rg_expression += r""
        #matches = re.findall(rg_expression,tweet,re.MULTILINE)
        try:
            matches = re.finditer(rg_expression,tweet,re.MULTILINE)
        except:
            answer = [x.strip("*?+") for x in answer]
            rg_expression = r"\b(" + "|".join(answer) + ")"
            for a in range(len(answer)):
                #rg_expression = rg_expression + "r(?:\W+\w+){0,6}?\W+(" + "|".join(answer)+ r")?"
                rg_expression = rg_expression + "r(?:\W+\w+){0,6}?\W{0,3}(" + "|".join(answer)+ r")?"
            rg_expression + r"\b"
            if answer:
                matches = re.finditer(rg_expression,tweet,re.MULTILINE)
            else:
                matches = None

        if matches:
            largest_match = 0
            m_group = None

            for m in matches:
                #print("match:{} count:{} span:{}".format(m.group(),len(m.group()),m.span()))
                if len(m.group()) > largest_match:
                    largest_match = len(m.group())
                m_group = m
            if m_group:
                data[idx]["answer_start"] = m_group.start()
                data[idx]["answer_length"] = m_group.end() - m_group.start()
                success.append(data[idx])
            else:
                data[idx]["rg_expression"] = rg_expression
                fails.append(data[idx])



    for idx in range(len(fails)):
        tweet = fails[idx]["Tweet"].lower()
        answer = fails[idx]["Answer"][0].lower().strip(".?+*")
        #matchs = re.match(answer,tweet)
        if answer in tweet:
            fails[idx]["answer_start"] = tweet.index(answer)
            fails[idx]["answer_length"] = len(answer)
            success2.append(answer)
        else:
            fails2.append(fails[idx])

    print(len(success))
    print(len(data))
    print(len(fails2))
    print(len(success2))
    print(len(success) + len(success2))
    for a in range(100):
        print(fails2[a])
        print()
    return success + success2

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
