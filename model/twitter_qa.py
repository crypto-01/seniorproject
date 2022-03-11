from transformers.models.bert  import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras import layers
import tensorflow as tf
from model.compute_f1_callback import Conputef1Callback
import numpy as np

class TwitterQa(object):

    def __init__(self,max_len,learning_rate):
        self.model = self.create_model(max_len,learning_rate)
        
    def create_model(self,max_len, learning_rate):
        max_len = max_len
        encoder = TFBertModel.from_pretrained("bert-base-uncased")
        input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
        embedding = encoder(
            input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]

        start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
        start_logits = layers.Flatten()(start_logits)

        end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
        end_logits = layers.Flatten()(end_logits)

        start_probs = layers.Activation(tf.keras.activations.softmax)(start_logits)
        end_probs = layers.Activation(tf.keras.activations.softmax)(end_logits)

        model = tf.keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[start_probs, end_probs],
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=[loss, loss])
        return model
    def train(self,train_batch,test_batch,epoch,batch_size):
        f1_score_callback = Conputef1Callback((test_batch.input_ids,test_batch.segment_ids,test_batch.mask_ids),test_batch.answers)
        #print(len(train_batch.input_ids),len(train_batch.segment_ids),len(train_batch.mask_ids))
        #self.model.fit((train_batch.input_ids[:50],train_batch.segment_ids[:50],train_batch.mask_ids[:50]),(train_batch.start_tokens_idx[:50],train_batch.end_tokens_idx[:50]),batch_size=batch_size,epochs=epoch,verbose=2,callbacks=[f1_score_callback])
        self.model.fit((train_batch.input_ids,train_batch.segment_ids,train_batch.mask_ids),(train_batch.start_tokens_idx,train_batch.end_tokens_idx),batch_size=batch_size,epochs=epoch,verbose=2,callbacks=[f1_score_callback])

    def predict(self,batch):
        pred_start, pred_end = self.model.predict((batch.input_ids,batch.segment_ids,batch.mask_ids))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        answers = []
        for idx,(start,end ,ids ) in enumerate(zip(pred_start,pred_end,batch.input_ids)):
            pred_answer_start = np.argmax(start)
            pred_answer_end = np.argmax(end)
            pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end
                +1])
            results_string = tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end
                    + 1]
            #print(batch.tweets[idx])
            #print(results_string)
            #print(pred_answer)
            #print()
            #print("Tweet: ",batch.tweets[idx])
            #print("Question :", batch.questions[idx])
            #print("Answer: ",pred_answer)
            #print()
            pred_answer = pred_answer.strip("##")
            if pred_answer == "":
                pass
                #pred_answer = batch.tweets[idx]
            batch_tweet = batch.tweets[idx]
            batch_question = batch.questions[idx]
            answers.append((batch_tweet,batch_question,pred_answer))
        return answers

    def save_weights(self):
        self.model.save_weights("twitterqamodelweights.h5")

    def load_weights(self):
        self.model.load_weights("twitterqamodelweights.h5")
