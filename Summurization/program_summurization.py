from summarizer import Summarizer, TransformerSummarizer
import torch
from rouge import Rouge
from evaluate import load
import json 

text_names = ["+The Origin of Species_Chapter 3",
              "A_Select_Party",
              "An_Old_Womans_Tale",
              "Arthur C Clarke - The Reluctant Orchid",
              "Clarke, Arthur C - Reluctant Orchid, The",
              "Passages_from_a_Relinquished",
              "Pratchett, Terry - Discworld ss - Troll Bridge",
              "Prayers_Written_at_Vailima",
              "Resnick, Mike - Dispatches",
              "Snow-Bound",
              "The_Adventure_of_the_Mazarin",
              "The_Witch_of_Atlas"]


def open_file(filename):
    file = open('texts/' + filename + ".txt", "r")
    text = file.read()
    file.close()
    print("File opened")
    return(text)

def generate_summary_bert(text):
    bert_model = Summarizer()
    bert_summary = ''.join(bert_model(text, min_length=10))
    print("Summary generated bert")
    return(bert_summary)

def generate_summary_xlnet(text):
    model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    full = ''.join(model(text, min_length=10))
    print("Summary generated xlnet")
    return(full)

def write_summary(filename, method, data):
    file = open('summaries/' + filename + '_' + method + '_summary.txt', 'w')
    file.write(data)
    file.close()
    print("File summary writted")

def analyze_rouge(reference_text, generated_text):
    rouge = Rouge()
    scores = rouge.get_scores(generated_text, reference_text)
    print("Rouge analyzed")
    return(scores)

def analyze_bert(reference_text, generated_text):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=[generated_text], references=[reference_text], model_type="distilbert-base-uncased")
    print("Bert analyzed")
    return(results)

def write_result_rouge(filename, method, data):
    file = open('results/' + filename + '_' + method + '_result_rouge.txt', 'w')
    for item in data:
        file.write(json.dumps(item))
    file.close()
    print("File result writted")

def write_result_bert(filename, method, data):
    file = open('results/' + filename + '_' + method + '_result_bert.txt', 'w')
    for key, value in data.items():
        file.write(json.dumps(key) + ': ' + json.dumps(value) + '\n')
    file.close()
    print("File result writted")


for name in text_names:
    print("Start " + name)

    text = open_file(name)

    #summary_bert
    summary = generate_summary_bert(text)
    write_summary(name, 'bert', summary)
    analyze_result = analyze_rouge(text, summary)
    write_result_rouge(name, 'bert', analyze_result)
    analyze_result = analyze_bert(text, summary)
    write_result_bert(name, 'bert', analyze_result)

    #summary_xlnet
    summary = generate_summary_xlnet(text)
    write_summary(name, 'xlnet', summary)
    analyze_result = analyze_rouge(text, summary)
    write_result_rouge(name, 'xlnet', analyze_result)
    analyze_result = analyze_bert(text, summary)
    write_result_bert(name, 'xlnet', analyze_result)

    print("Finish " + name)
    

    

