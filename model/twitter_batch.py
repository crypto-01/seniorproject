
from transformers.models.bert  import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import numpy as np


class TwitterBatch():
    tokenizer = BertWordPieceTokenizer("/home/crypto/Programming/seniorproject/SeniorDesign2021/model/bert_base_uncased/vocab.txt",
            lowercase=True)
    def __init__(self, questions, tweets, answers,answers_start,answers_length):
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

    def preprocess(self):
        """process trainig data before training"""
        #tokenize both question and tweet
        self.inputs_encoding = list(map(self.tokenizer.encode, self.tweets,self.questions))
        self.input_ids = [inputs_id_list.ids for inputs_id_list in
                self.inputs_encoding]
        #turns inputs_ids which are numbers to characters
        #self.tokens = list(map(self.tokenizer.convert_ids_to_tokens, self.input_ids))
        self.tokens = [tks.tokens for tks in self.inputs_encoding ]
        #find end of each tweet and length of each tokens
        end_of_tweet_and_length = list(map(lambda x: (x.index("[SEP]"), len(x)), self.tokens))
        #show where if a token belongs to a tweet or question
        self.segment_ids = list(map(lambda x: [0] * (x[0] + 1) + [1] * (x[1] - x[0] -1 ), end_of_tweet_and_length))
        #used to show where there is padding and where there is content
        self.mask_ids = list(map(lambda x: [1] * (x[1]),end_of_tweet_and_length))


        #find start and end of answer tokens in tweet
        if self.answers:
            for data in range(self.batch_length):
                ans_token_idx = []
                for idx, (start, end) in enumerate(self.inputs_encoding[data].offsets):
                    if(self.tokens[data][idx] == "[SEP]"): break
                    if( start >= self.answers_start[data] and end <=
                            self.answers_start[data] +
                            self.answers_length[data]) and self.tokens[data][idx] !="[CLS]":
                        ans_token_idx.append(idx)
                if len(ans_token_idx)== 0:
                    self.start_tokens_idx.append(0)
                    self.end_tokens_idx.append(0)
                else:
                    self.start_tokens_idx.append(ans_token_idx[0])
                    self.end_tokens_idx.append(ans_token_idx[-1])

        #create padding for inputs_ids, segment_ids , and mask_ids
        #padding_length = (max(end_of_tweet_and_length, key=lambda x: x[1]))[1]
        padding_length = 200
        for data in range(self.batch_length):
           self.input_ids[data] = self.input_ids[data] + ([0] * (padding_length - len(self.inputs_encoding[data])))
           self.segment_ids[data] = self.segment_ids[data] + ([0] * (padding_length - len(self.segment_ids[data])))
           self.mask_ids[data] = self.mask_ids[data] + ([0] * (padding_length - len(self.mask_ids[data])))
           if self.answers:
               self.start_tokens_idx[data] = np.array([self.start_tokens_idx[data]])
               self.end_tokens_idx[data] = np.array([self.end_tokens_idx[data]])

        self.input_ids = list(map(np.array,self.input_ids))
        self.segment_ids = list(map(np.array,self.segment_ids))
        self.mask_ids = list(map(np.array,self.mask_ids))
        self.input_ids = np.array(self.input_ids)
        self.segment_ids = np.array(self.segment_ids)
        self.mask_ids = np.array(self.mask_ids)
        self.start_tokens_idx = np.array(self.start_tokens_idx)
        self.end_tokens_idx = np.array(self.end_tokens_idx)
        #self.start_tokens_idx = list(map(np.array,self.start_tokens_idx))
        #self.start_tokens_idx = list(map(lambda x: np.array([x]),self.start_tokens_idx))
        #self.end_tokens_idx = list(map(lambda x: np.array(x),self.end_tokens_idx))

