import unittest
from model.twitter_batch import TwitterBatch
from model.func_utility import find_answer_in_tweet,create_batche
import numpy as np
import os
import json

class TestTwiterrBatch(unittest.TestCase):
    def test_len_of_model_inputs_after_preprocess(self):
        TRAIN_PATH = "TweetQA_data/train.json"
        with open(TRAIN_PATH) as f:
            data = json.load(f)
        data =find_answer_in_tweet(data=data)
        train_batch = create_batche(data[:10])
        #train_batch.segment_ids =train_batch.segment_ids[:2]
        self.assertEqual(len(train_batch.input_ids),10)
        self.assertEqual(len(train_batch.segment_ids),10)
        self.assertEqual(len(train_batch.mask_ids),10)
    def test_inputs_not_equal(self):
        TRAIN_PATH = "TweetQA_data/train.json"
        with open(TRAIN_PATH) as f:
            data = json.load(f)
        data =find_answer_in_tweet(data=data)
        train_batch = create_batche(data[:10])
        #train_batch.mask_ids = train_batch.input_ids
        self.assertEqual(np.array_equal(train_batch.input_ids,train_batch.mask_ids),False)
    def test_no_empy_sample_in_batch(self):
        TRAIN_PATH = "TweetQA_data/train.json"
        with open(TRAIN_PATH) as f:
            data = json.load(f)
        data =find_answer_in_tweet(data=data)
        train_batch = create_batche(data[:10])
        self.assertNotEqual(len(train_batch.input_ids[0]),0)

        
    
