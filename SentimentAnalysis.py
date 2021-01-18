#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flask
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from nltk.tokenize import sent_tokenize
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = False

def sentiment_scores(sentence): 
    sid_obj = SentimentIntensityAnalyzer() 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    return sentiment_dict
@app.route('/', methods=['GET'])
def sentiments():
    text_2000= request.args.get("utext")
    text = text_2000
    textl = sent_tokenize(text)
    eachInASeparateLine = "\n".join(sent_tokenize(text))
    res = []
    res.append({"full_text":textl})
    for t in textl:
        res.append(sentiment_scores(t))
    return str(res)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4800)

