from transformers.models.bert  import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from model.compute_f1_callback import Conputef1Callback
from model.func_utility import get_class_weights
import numpy as np

class TwitterQa(object):

    def __init__(self,max_len,learning_rate,weights=None,load_model=False):
        #self.model = self.create_model(max_len,learning_rate)
        if not load_model:
            self.model = self.create_model(max_len,learning_rate,0,100,weights)
        else:
            self.load_model()
        
    def create_model(self,max_len, learning_rate,n_hidden_layers,n_nodes,weights=None):
        max_len = max_len
        encoder = TFBertModel.from_pretrained("bert-base-uncased")
        #encoder.trainable = False
        input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
        embedding = encoder.bert(
            input_ids = input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]

        #start_logits = embedding
        #end_logits = embedding
        #start_logits = layers.Dropout(.5)(start_logits)
        #end_logits = layers.Dropout(.5)(end_logits)
        logits_hidden = embedding
        #logits_hidden = layers.Dropout(.2)(logits_hidden)
        for a in range(n_hidden_layers):
            #start_logits = layers.Dense(n_nodes, name="start_logit_hidden"+ str(a), use_bias=True,kernel_regularizer=regularizers.l2(.01))(start_logits)
            #start_logits = layers.Dense(n_nodes, name="start_logit_hidden"+ str(a), use_bias=False)(start_logits)
            ##start_logits = layers.Dense(n_nodes,  use_bias=False)(start_logits)
            ##start_logits = layers.BatchNormalization()(start_logits)
            ##start_logits = layers.Activation(tf.keras.activations.relu)(start_logits)
            #start_logits = layers.Activation(tf.keras.activations.sigmoid)(start_logits)
            ##start_logits = layers.Dropout(.5)(start_logits)
            logits_hidden = layers.Dense(n_nodes,  use_bias=True)(logits_hidden)
            #logits_hidden = layers.BatchNormalization()(logits_hidden)
            logits_hidden = layers.Activation(tf.keras.activations.relu)(logits_hidden)
            #logits_hidden = layers.Activation(tf.keras.activations.sigmoid)(logits_hidden)
            #logits_hidden = layers.Dropout(.1)(logits_hidden)
            #end_logits = layers.Dense(n_nodes, name="end_logit_hidden" + str(a), use_bias=True,kernel_regularizer=regularizers.l2(.01))(end_logits)
            #end_logits = layers.Dense(n_nodes, name="end_logit_hidden" + str(a), use_bias=False)(end_logits)
            ##end_logits = layers.Dense(n_nodes, use_bias=False)(end_logits)
            ##end_logits = layers.BatchNormalization()(end_logits)
            ##end_logits = layers.Activation(tf.keras.activations.relu)(end_logits)
            #end_logits = layers.Activation(tf.keras.activations.sigmoid)(end_logits)
            ##end_logits = layers.Dropout(.5)(end_logits)


        start_logits = layers.Dense(1,name="start_logit")(logits_hidden)
        #start_logits = layers.Dense(1,name="start_logit")(start_logits)
        start_logits = layers.Flatten()(start_logits)
        #start_logits = layers.Dropout(.2)(start_logits)
        end_logits = layers.Dense(1,name="end_logit")(logits_hidden)
        #end_logits = layers.Dense(1,name="end_logit")(end_logits)
        end_logits = layers.Flatten()(end_logits)
        #end_logits = layers.Dropout(.2)(end_logits)
        start_probs = layers.Activation(tf.keras.activations.softmax)(start_logits)
        end_probs = layers.Activation(tf.keras.activations.softmax)(end_logits)
        #start_probs = start_logits
        #end_probs = end_logits

        #optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
        #optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate,momentum=.9,nesterov=True,clipvalue=1)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        if weights:
            model = tf.keras.Model(
                inputs=[input_ids, token_type_ids, attention_mask],
                outputs=[end_logits, end_logits],
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss_start = self.weighted_categorical_crossentropy(list(weights[0].values()))
            loss_end = self.weighted_categorical_crossentropy(list(weights[1].values()))
            model.compile(optimizer=optimizer, loss=[loss_start, loss_end],metrics=["accuracy"])
        else:
            model = tf.keras.Model(
                inputs=[input_ids, token_type_ids, attention_mask],
                outputs=[start_probs, end_probs],
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=[loss, loss],metrics=["accuracy"])

        #model.compile(optimizer=optimizer, loss=[loss_start, loss_end],metrics=["accuracy"])
        model.summary()
        return model
    def train(self,train_batch,test_batch,epoch,batch_size):
        if not self.model:
            return
        f1_score_callback = Conputef1Callback(test_batch)
        #print(len(train_batch.input_ids),len(train_batch.segment_ids),len(train_batch.mask_ids))
        #self.model.fit((train_batch.input_ids[:50],train_batch.segment_ids[:50],train_batch.mask_ids[:50]),(train_batch.start_tokens_idx[:50],train_batch.end_tokens_idx[:50]),batch_size=batch_size,epochs=epoch,verbose=2,callbacks=[f1_score_callback])
        #self.model.fit((train_batch.input_ids,train_batch.segment_ids,train_batch.mask_ids),(train_batch.start_tokens_idx,train_batch.end_tokens_idx),batch_size=batch_size,epochs=epoch,verbose=2,callbacks=[f1_score_callback])
        self.model.fit((train_batch.input_ids,train_batch.segment_ids,train_batch.mask_ids),(train_batch.start_tokens_idx,train_batch.end_tokens_idx),batch_size=batch_size,epochs=epoch,verbose=1,validation_data=((test_batch.input_ids,test_batch.segment_ids,test_batch.mask_ids),(test_batch.start_tokens_idx,test_batch.end_tokens_idx)),callbacks=[f1_score_callback],shuffle=True)
        #self.model.fit((train_batch.input_ids,train_batch.segment_ids,train_batch.mask_ids),(train_batch.start_tokens_idx,train_batch.end_tokens_idx),batch_size=batch_size,epochs=epoch,verbose=2,validation_data=((test_batch.input_ids,test_batch.segment_ids,test_batch.mask_ids),(test_batch.start_tokens_idx,test_batch.end_tokens_idx)),shuffle = True,class_weight={"start":class_weight_start,"end":class_weight_end})
    def weighted_categorical_crossentropy(self,class_weight):
        def loss(y_true, y_pred):
            y_true = tf.dtypes.cast(y_true, tf.int32)
            onehot = tf.one_hot(tf.reshape(y_true,[-1]), depth=len(class_weight))
            weight = tf.math.multiply(class_weight, onehot)
            weight = tf.reduce_sum(weight, axis = -1)
            losses = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=y_true,logits=y_pred,weights=weight)
            return losses
        return loss

    def predict(self,batch):
        if not self.model:
            return
        pred_start, pred_end = self.model.predict((batch.input_ids,batch.segment_ids,batch.mask_ids))
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        answers = []
        for idx,(start,end ,tweet,span) in enumerate(zip(pred_start,pred_end,batch.tweets,batch.spans)):
            pred_answer_start = np.argmax(start)
            pred_answer_end = np.argmax(end)
            #pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end ])
            pred_answer = ""
            if pred_answer_start < len(span):
                original_start_char = span[pred_answer_start][0]
                if pred_answer_end < len(span):
                    original_end_char = span[pred_answer_end][1]
                    pred_answer = tweet[original_start_char:original_end_char]
                else:
                    pred_answer = tweet[original_start_char:]

            if pred_answer.startswith("[CLS]"):
                pred_answer = pred_answer[5:]
            pred_answer = pred_answer.strip("##")
            if pred_answer.startswith("[CLS]"):
                pred_answer = pred_answer[5:]
            pred_answer = pred_answer.strip("##")
            if pred_answer == "":
                #pass
                #pred_answer = batch.tweets[idx]
                pred_answer = tweet
            batch_tweet = batch.tweets[idx]
            batch_question = batch.questions[idx]
            answers.append((batch_tweet,batch_question,pred_answer))
        return answers

    def save_weights(self,weigt_location = "twitterqamodelweights.h5"):
        if self.model:
            self.model.save_weights(weigt_location)

    def load_weights(self,weigt_location="twitterqamodelweights.h5"):
        if self.model:
            self.model.load_weights(weigt_location)
    def save_model(self,file_location="saved_model"):
        if self.model:
            self.model.save(file_location)
    def load_model(self,file_location="saved_model"):
        self.model = tf.keras.models.load_model(file_location)
        #self.model = tf.saved_model.load(file_location)
