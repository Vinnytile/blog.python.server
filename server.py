from flask import Flask, request
from flask_cors import CORS
from summarizer import Summarizer, TransformerSummarizer

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
