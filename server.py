from flask import Flask, request
from flask_cors import CORS

from summarizer import Summarizer, TransformerSummarizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import spacy
import json

app = Flask(__name__)
CORS(app)

@app.route("/summarize", methods=['POST'])
def summarize_text():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        summary = summarize(json["text"])
        return summary
    else:
        return 'Content-Type not supported!'

def summarize(text):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(text, min_length=60))
    return(bert_summary)


@app.route("/sentiments", methods=['POST'])
def analyze_sentiment_text():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        sentiments = analyze_sentiment(json["text"])
        return sentiments
    else:
        return 'Content-Type not supported!'

def analyze_sentiment(text):
    sentiment = SentimentIntensityAnalyzer()
    sent = sentiment.polarity_scores(text)
    return(sent)


@app.route("/namedentities", methods=['POST'])
def named_entities_text():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        sentiments = named_entities(json["text"])
        return sentiments
    else:
        return 'Content-Type not supported!'

def named_entities(text):
    NER = spacy.load("en_core_web_sm")
    entities_text = NER(text)

    entities = []
    for ent in entities_text.ents:
        entities.append((ent.text, ent.label_))

    jsonStr = json.dumps(dict(entities))
    return(jsonStr)