import string
import re
#from twitter_batch import TwitterBatch
from model.twitter_batch import *
from sklearn.utils.class_weight import compute_class_weight
import collections



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

def get_class_distribution_with_json(data):
    class_distribution = {"answer_start":{}}
    n_numbe_of_prints = 0
    if "answer_start" not in data[0].keys():
        return {}
    for tweet in data:
        start_pos = tweet["answer_start"]
        if class_distribution["answer_start"].get(start_pos):
            class_distribution["answer_start"][start_pos] +=1
            """if start_pos == 0 and n_numbe_of_prints < 20:
                print(tweet["Tweet"])
                print(tweet["Answer"])
                print()
                """
        else:
            class_distribution["answer_start"][start_pos] =1
            """
            if start_pos == 0 and n_numbe_of_prints < 20:
                print(tweet["Tweet"])
                print(tweet["Answer"])
                print()
                """
    return class_distribution
def get_class_weights(data,n_class):
    class_labels = np.unique(data).tolist()
    data = data.tolist()
    print(class_labels)
    for a in range(n_class):
        if a not in class_labels:
            class_labels.append(a)
            data.append(a)
            
    class_labels.sort()
    #class_labels = [*range(n_class)]
    #print(class_labels)
    #print(data[:100])
    #print(max(data))
    class_weights = compute_class_weight(class_weight='balanced',classes=class_labels,y=data)
    #class_weights = dict((x,y) for x, y in zip(unique,counts))
    class_weights_dic = dict(zip(class_labels,class_weights))
    #for a in range(n_class):
    #    if class_weights_dic.get(a) != None:
    #        class_weights_dic[a] = 0

    return class_weights_dic

    



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
def find_answer_in_tweet2(data):
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
                valid_groups = len(m.groups()) - m.groups().count(None)
                #if len(m.group()) > largest_match:
                    #largest_match = len(m.group())
                    #m_group = m
                if valid_groups > largest_match:
                    largest_match = valid_groups
                    m_group = m
            if m_group:
                valid_groups = len(m_group.groups()) - m_group.groups().count(None)
                #n_not_null = len(matched_groups) - matched_groups.count("")
                #if len(m_group.group()) ==1:
                if valid_groups == 1:
                    answer_capture = m_group.group(0)
                    # Remove articles
                    regex = re.compile(r"\b(a|an|the|he|she|on|it|is|in|to|her|us|or)\b", re.UNICODE)
                    answer_capture = re.sub(regex, " ", answer_capture)
                    """
                    if None in m_group.groups():
                        fo = m_group.groups.index(None) 
                        answer_length = m_group.end(fo) - m_group.start(0)
                    else:
                        answer_length = m_group.end(-1) - m_group.start(0)
                        """

                    if len(answer_capture) > 1:
                        data[idx]["answer_start"] = m_group.start()
                        #data[idx]["answer_length"] = m_group.end() - m_group.start() -1
                        if m_group.lastindex:
                            data[idx]["answer_length"] = m_group.end(m_group.lastindex) - m_group.start(1) 
                        else:
                            data[idx]["answer_length"] = m_group.end() - m_group.start() -1
                        data[idx]["groups"]= m_group.groups()
                        data[idx]["group_text"]= m_group.group(0)
                        success.append(data[idx])
                    else:
                        data[idx]["rg_expression"] = rg_expression
                        fails.append(data[idx])

                else:
                    data[idx]["answer_start"] = m_group.start()
                    #data[idx]["answer_length"] = m_group.end() - m_group.start() -1
                    if m_group.lastindex:
                        data[idx]["answer_length"] = m_group.end(m_group.lastindex) - m_group.start(1) 
                    else:
                        data[idx]["answer_length"] = m_group.end() - m_group.start() -1
                    data[idx]["groups"]= m_group.groups()
                    data[idx]["group_text"]= m_group.group(0)
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
            data[idx]["groups"]= answer
            data[idx]["group_text"]= answer
            success2.append(fails[idx])
        else:
            fails2.append(fails[idx])

    return success + success2

def process_squid_data(data):
    data_processed = []
    #all_data = data.get("data")
    all_data = data
    for d in all_data:
        paragraphs = d.get("paragraphs")
        for p in paragraphs:
            context = p.get("context")
            qas = p.get("qas")
            for q in qas:
                tweet = {}
                tweet["Question"]= q.get("question")
                tweet["Answer"] = [q.get("answers")[0].get("text")]
                tweet["Tweet"] = context
                tweet["qid"] = q.get("id")
                tweet["answer_start"] = q.get("answers")[0].get("answer_start")
                tweet["answer_length"] = len(q.get("answers")[0].get("text"))
                data_processed.append(tweet)
    return data_processed
def get_tokesn(text):
    if not text: return []
    return normalize_text(text).split() 

def compute_f1(true,pred):

    true_tokens = normalize_text(true).split()
    pred_tokens = normalize_text(pred).split()
    #true positive
    common = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
    num_of_same_tokens = sum(common.values())
    if len(true_tokens) == 0 or len(pred_tokens) == 0:
        return int(true_tokens == pred_tokens)
    if num_of_same_tokens == 0:
        return 0
    precision = num_of_same_tokens / len(pred_tokens)
    recall = num_of_same_tokens / len(true_tokens)
    #tp = num_of_same_tokens
    #fp = len(pred_tokens) - num_of_same_tokens
    #fn = len(true_tokens) - num_of_same_tokens
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_confusion_and_f1_score(true,pred):
    true_tokens = normalize_text(true).split()
    pred_tokens = normalize_text(pred).split()
    #true positive
    common = collections.Counter(true_tokens) & collections.Counter(pred_tokens)
    num_of_same_tokens = sum(common.values())
    #if len(true_tokens) == 0 or len(pred_tokens) == 0:
        #return int(true_tokens == pred_tokens)
    #if num_of_same_tokens == 0:
    #    return 0
    #if len(true_tokens) == 0 or len(pred_tokens) == 0:
    #precision = num_of_same_tokens / len(pred_tokens)
    #recall = num_of_same_tokens / len(true_tokens)
    if len(true_tokens) == 0 or len(pred_tokens) == 0:
        precision = 0
        recall = 0
    else:
        precision = num_of_same_tokens / len(pred_tokens)
        recall = num_of_same_tokens / len(true_tokens)


    tp = num_of_same_tokens
    fp = len(pred_tokens) - num_of_same_tokens
    fn = len(true_tokens) - num_of_same_tokens
    if precision and recall:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    return [[precision,recall,f1], [tp,fp,fn]]
    #return (precision,recall,f1)
    #return tp, f1;


def compute_exact_match(true,pred):
    return int(normalize_text(true) == normalize_text(pred))

    
def create_batche(data):
    tweets = [item.get("Tweet").lower() for item in data]
    questions = [item.get("Question").lower() for item in data]
    answers_qid = [item.get("qid") for item in data]
    if "Answer" in data[0].keys():
        answers = [item.get("Answer")[0].lower() for item in data]
        answers_start = [item.get("answer_start") for item in data]
        answers_length = [item.get("answer_length") for item in data]
    else:
        answers = []
        answers_start = []
        answers_length = []
    
    batch = TwitterBatch(questions,tweets,answers,answers_start,answers_length,answers_qid)
    batch.preprocess()
    return batch
