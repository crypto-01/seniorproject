
from transformers.models.bert  import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import numpy as np


class TwitterBatch():
    tokenizer = BertWordPieceTokenizer("model/bert_base_uncased/vocab.txt",
            lowercase=True)
    def __init__(self, questions, tweets, answers,answers_start,answers_length,qid):
        self.questions = questions
        self.tweets = tweets
        self.answers = answers
        self.answers_start = answers_start
        self.answers_length = answers_length
        self.batch_length = len(self.questions)
        self.start_tokens_idx =[] #[0] * self.batch_length
        self.end_tokens_idx = [] #[0] * self.batch_length
        self.input_ids = []
        self.tokens = []
        self.segment_ids = []
        self.mask_ids = []
        self.inputs_encoding=[]
        self.spans = []
        self.qid = qid

    def preprocess(self):
        """process trainig data before training"""
        #tokenize both question and tweet
        self.inputs_encoding = list(map(self.tokenizer.encode, self.tweets,self.questions))
        self.input_ids = [inputs_id_list.ids for inputs_id_list in
                self.inputs_encoding]
        #turns inputs_ids which are numbers to characters
        #self.tokens = list(map(self.tokenizer.convert_ids_to_tokens, self.input_ids))
        self.tokens = [tks.tokens for tks in self.inputs_encoding ]
        self.tokens = [tks.tokens for tks in self.inputs_encoding ]
        #save original location of characters
        self.spans = [sp.offsets for sp in self.inputs_encoding]
        #find end of each tweet and length of each tokens
        end_of_tweet_and_length = list(map(lambda x: (x.index("[SEP]"), len(x)), self.tokens))
        #show where if a token belongs to a tweet or question
        self.segment_ids = list(map(lambda x: [0] * (x[0] + 1) + [1] * (x[1] - x[0] -1 ), end_of_tweet_and_length))
        #used to show where there is padding and where there is content
        self.mask_ids = list(map(lambda x: [1] * (x[1]),end_of_tweet_and_length))


        #find start and end of answer tokens in tweet
        start_tokens_idx_temp = []
        end_tokens_idx_temp = []
        errors_found = 0
        tweets_to_remove = []
        if self.answers:
            #for data in range(self.batch_length):
            for data in range(len(self.tweets)):
                ans_token_idx = []
                for idx, (start, end) in enumerate(self.inputs_encoding[data].offsets):
                    if(self.tokens[data][idx] == "[SEP]"): break
                   # if( start >= self.answers_start[data] and end <= self.answers_start[data] + self.answers_length[data]): 
                   #     ans_token_idx.append(idx)
                    if( start >= self.answers_start[data] and start <= self.answers_start[data] + self.answers_length[data]): 
                        ans_token_idx.append(idx)
                    elif end >= self.answers_start[data] and end <= self.answers_start[data] + self.answers_length[data]:
                        ans_token_idx.append(idx)
                    #elif self.tokens[data][idx] !="[CLS]":
                        #print("errors")
                        #print(self.tokens[data][idx],start,self.answers_start[data])
                if len(ans_token_idx)== 0:
                    #self.start_tokens_idx.append(0)
                    #self.end_tokens_idx.append(0)
                    #tweets_to_remove.append(data)
                    start_tokens_idx_temp.append(0)
                    end_tokens_idx_temp.append(0)
                    tweets_to_remove.append(data)
                    errors_found +=1
                    #print("errors")
                    #print(self.tweets[data])
                    #print(self.tokens[data])
                    #print(self.answers[data])
                    #print(self.tweets[data][self.answers_start[data]:self.answers_start[data] + self.answers_length[data]])
                    #print()
                    #print(self.inputs_encoding[data].offsets)
                    #print(self.inputs_encoding[data].tokens)
                    #print(self.answers_start[data],self.answers_length[data])
                    #print(data)
                else:
                    #self.start_tokens_idx.append(ans_token_idx[0])
                    #self.end_tokens_idx.append(ans_token_idx[-1])
                    start_tokens_idx_temp.append(ans_token_idx[0])
                    end_tokens_idx_temp.append(ans_token_idx[-1])

        #create padding for inputs_ids, segment_ids , and mask_ids
        #padding_length = (max(end_of_tweet_and_length, key=lambda x: x[1]))[1]
        print(errors_found)
        padding_length = 200
        """for data in range(self.batch_length):
            if (padding_length - len(self.input_ids[data])) < 0:
                print("errors")
            #self.input_ids[data] = self.input_ids[data] + ([0] * (padding_length - len(self.inputs_encoding[data])))
            self.input_ids[data] = self.input_ids[data] + ([0] * (padding_length - len(self.input_ids[data])))
            self.segment_ids[data] = self.segment_ids[data] + ([0] * (padding_length - len(self.segment_ids[data])))
            self.mask_ids[data] = self.mask_ids[data] + ([0] * (padding_length - len(self.mask_ids[data])))
           #if self.answers:
           #    self.start_tokens_idx[data] = np.array([self.start_tokens_idx[data]])
           #    self.end_tokens_idx[data] = np.array([self.end_tokens_idx[data]])
           """
        for data in range(len(self.input_ids)-1,-1,-1):
            if (padding_length - len(self.input_ids[data])) < 0:
                del self.input_ids[data]
                del self.segment_ids[data]
                del self.mask_ids[data]
                del start_tokens_idx_temp[data]
                del end_tokens_idx_temp[data]
                del self.qid[data]
                del self.tweets[data]
                del self.questions[data]
                del self.answers[data]
                del self.tokens[data]
                del self.spans[data]
                self.batch_length -=1
            elif data in tweets_to_remove:
                del self.input_ids[data]
                del self.segment_ids[data]
                del self.mask_ids[data]
                del start_tokens_idx_temp[data]
                del end_tokens_idx_temp[data]
                del self.qid[data]
                del self.tweets[data]
                del self.questions[data]
                del self.answers[data]
                del self.tokens[data]
                del self.spans[data]
                self.batch_length -=1

            #self.input_ids[data] = self.input_ids[data] + ([0] * (padding_length - len(self.inputs_encoding[data])))
            else:
                self.input_ids[data] = self.input_ids[data] + ([0] * (padding_length - len(self.input_ids[data])))
                self.segment_ids[data] = self.segment_ids[data] + ([0] * (padding_length - len(self.segment_ids[data])))
                self.mask_ids[data] = self.mask_ids[data] + ([0] * (padding_length - len(self.mask_ids[data])))

        self.input_ids = list(map(np.array,self.input_ids))
        self.segment_ids = list(map(np.array,self.segment_ids))
        self.mask_ids = list(map(np.array,self.mask_ids))
        self.input_ids = np.array(self.input_ids)
        self.segment_ids = np.array(self.segment_ids)
        self.mask_ids = np.array(self.mask_ids)
        #self.start_tokens_idx = np.array(self.start_tokens_idx)
        #self.end_tokens_idx = np.array(self.end_tokens_idx)
        if start_tokens_idx_temp:
            self.start_tokens_idx = np.array(start_tokens_idx_temp)
            self.end_tokens_idx = np.array(end_tokens_idx_temp)

        #self.start_tokens_idx = list(map(np.array,self.start_tokens_idx))
        #self.start_tokens_idx = list(map(lambda x: np.array([x]),self.start_tokens_idx))
