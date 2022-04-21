from transformers.models.bert.tokenization_bert  import BertTokenizer
from tensorflow import keras
import numpy as np
from model.func_utility import normalize_text
from model.func_utility import compute_f1
from model.func_utility import compute_exact_match

class Conputef1Callback(keras.callbacks.Callback):
    #tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def __init__(self,batch):
        """uses the f-score with precision recall tp fn and fp"""
        #self.x_eval = x_eval
        #self.y_eval = y_eval
        self.batch = batch


    """
    def on_epoch_end(self, epoch, logs=None):
        f1_scores =[]
        #pred_start, pred_end = self.model.predict((self.x_eval[0][:50],self.x_eval[1][:50],self.x_eval[2][:50]))
        #pred_start, pred_end = self.model.predict((self.x_eval[0][:],self.x_eval[1][:],self.x_eval[2][:]))
        pred_start, pred_end = self.model.predict((self.batch.input_ids,self.batch.segment_ids,self.batch.mask_ids))
        #for start, end, ids ,answer in zip(pred_start, pred_end,
                #self.x_eval[0][:50],self.y_eval[:50]):
        for start, end, ids ,answer in zip(pred_start, pred_end,
                self.x_eval[0][:],self.y_eval[:]):
            pred_answer_start = np.argmax(start)
            pred_answer_end = np.argmax(end)
            pred_answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end])
            pred_answer_tokens = normalize_text(pred_answer).split()
            answer_tokens = normalize_text(answer).split()
            if (len(pred_answer_tokens) == 0 or len(answer_tokens) == 0):
                f1_scores.append(0)
                continue
            common_tokens = set(pred_answer_tokens) & set(answer_tokens)
            if (len(common_tokens) == 0):
                f1_scores.append(0)
                continue
            prec = len(common_tokens) / len(pred_answer_tokens)
            rec = len(common_tokens) / len(answer_tokens)
            f1_scores.append(2 * (prec * rec) / (prec + rec))
        average_f1_score = sum(f1_scores) / len(f1_scores)
        print(f"\nepoch={epoch+1}, f1 avg score={average_f1_score:.2f}")
    #def on_train_batch_end(self, batch, logs=None):
        #keys = list(logs.values())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
       """     
    def on_epoch_end(self, epoch, logs=None):
        f1_scores =[]
        exact_score = 0
        batch = self.batch
        pred_start, pred_end = self.model.predict((batch.input_ids,batch.segment_ids,batch.mask_ids))
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for idx,(start,end ,ids,answer ) in enumerate(zip(pred_start,pred_end,batch.input_ids,batch.answers)):
            pred_answer_start = np.argmax(start)
            pred_answer_end = np.argmax(end)
            pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end ])
            #results_string = tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end ]
            #pred_answer = results_string
            #print(batch.tweets[idx])
            #print(results_string)
            #print(pred_answer)
            #print()
            #print("Tweet: ",batch.tweets[idx])
            #print("Question :", batch.questions[idx])
            #print("Answer: ",pred_answer)
            #print()
            if pred_answer.startswith("[CLS]"):
                pred_answer = pred_answer[5:]
            pred_answer = pred_answer.strip("##")
            if pred_answer == "":
                #pass
                pred_answer = batch.tweets[idx]
            f1_score = compute_f1(answer,pred_answer)
            f1_scores.append(f1_score)
            """
        for idx,(start,end,tweet,answer,span ) in enumerate(zip(pred_start,pred_end,batch.tweets,batch.answers,batch.spans)):
            pred_answer_start = np.argmax(start)
            pred_answer_end = np.argmax(end)
            #pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end ])
            #results_string = tokenizer.convert_ids_to_tokens(ids)[pred_answer_start:pred_answer_end ]
            #pred_answer = results_string
            #print(batch.tweets[idx])
            #print(results_string)
            #print(pred_answer)
            #print()
            #print("Tweet: ",batch.tweets[idx])
            #print("Question :", batch.questions[idx])
            #print("Answer: ",pred_answer)
            #print()
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
            if pred_answer == "":
                pass
                #pred_answer = batch.tweets[idx]
            f1_score = compute_f1(answer,pred_answer)
            exact_score += compute_exact_match(answer,pred_answer)
            f1_scores.append(f1_score)
        average_f1_score = sum(f1_scores) / len(f1_scores)
        exact_score = exact_score / batch.batch_length
        print(f"\nepoch={epoch+1}, f1 avg score={average_f1_score:.2f}, ex_score={exact_score:.2f}")

