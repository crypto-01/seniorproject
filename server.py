from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import json
from model.func_utility import find_answer_in_tweet ,create_batche
from model.twitter_qa import TwitterQa
from model.tweetqa_eval import *
import random

import os

os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
twitterqa = TwitterQa(200,5e-5)
twitterqa.load_weights()
test_data_loc = "./TweetQA_data/test.json"
with open(test_data_loc) as f:
    test_data = json.load(f)

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        data = request.json
        if data == None:
            model_response = {}
            model_response["response"] = "can not answer that"
            return model_response
        tweet = data["tweet"]
        question = data["question"]
        data = [{"Tweet":tweet,"Question":question}]
        #answer = "model response"
        batch = create_batche(data)
        answers = twitterqa.predict(batch)
        if(len(answers[0][2]) == 0):
            answers[0][2] = "can not answer that question"
        
        model_response = {
                "response": answers[0][2]
                }
        #return render_template("index.html",model_response=answer)
        #return ("",204)
        return model_response
    else:
        return render_template("index.html")
@app.route("/getrandomtweet",methods=["GET"])
def get_random_tweet():
    if request.method == "GET":
        tweet_choice = random.choice(test_data)
        random_tweet = {}
        #random_tweet["tweet"] = test_data[0]["Tweet"]
        #random_tweet["question"] = test_data[0]["Question"]
        random_tweet["tweet"] = tweet_choice["Tweet"]
        random_tweet["question"] = tweet_choice["Question"]
        return random_tweet
@app.route("/dataset" ,methods=["POST","GET"])
def dataset():
    if request.method == "GET":
        return render_template("datasetfile.html")
    else:
        #print("posted file: {}".format(request.files['avatar']))
        file  = request.files['avatar']
        #file = request.form["avatar"]
        if file:
            file.seek(0)
            data = file.read()
            data = json.loads(data)
            data =find_answer_in_tweet(data=data)
            #testing_data_length = int(len(data) * .06)
            testing_data_length = 50
            #testing_data_length = 400
            #testing_data_length = int(len(data))
            #batch = create_batche(data[:data_used])
            batch = create_batche(data[-testing_data_length:])
            #twitterqa = TwitterQa(200,5e-5)
            #twitterqa.load_weights()
            predictions = twitterqa.predict(batch)
            #print()
            #print("Tweet: ",predictions[9][0])
            #print("Question: ",predictions[9][1])
            #print("Answer: ",predictions[9][2])
            #print()
            
            with open("gold_file.json","w") as f:
                json.dump(data[-testing_data_length:],f)
            for a in range(testing_data_length):
                data[-testing_data_length + a]["Answer"] = [predictions[a][2]]
            with open("prediction_file.json","w") as f:
                json.dump(data[-testing_data_length:],f)

            scores = evaluate("gold_file.json","prediction_file.json","test")
            #response {"response": scores}
            return scores["result"][0]
        #print(file)
        #return render_template("datasetfile.html")
        return "score could not be conputed"

if __name__ == "__main__":
    app.debug = True
    app.run(host = "0.0.0.0")
