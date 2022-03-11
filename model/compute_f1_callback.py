from transformers.models.bert.tokenization_bert  import BertTokenizer
from tensorflow import keras
import numpy as np
from model.func_utility import normalize_text

class Conputef1Callback(keras.callbacks.Callback):
    #tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def __init__(self, x_eval,y_eval):
        """uses the f-score with precision recall tp fn and fp"""
        self.x_eval = x_eval
        self.y_eval = y_eval


    def on_epoch_end(self, epoch, logs=None):
        f1_scores =[]
        pred_start, pred_end = self.model.predict((self.x_eval[0][:50],self.x_eval[1][:50],self.x_eval[2][:50]))
        for start, end, ids ,answer in zip(pred_start, pred_end,
                self.x_eval[0][:50],self.y_eval[:50]):
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
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.values())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
            
