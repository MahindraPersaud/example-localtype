from flask import render_template
from localtype import app
from flask import request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

import dill

import localtype.synonyms as syn
import localtype.lime_custom_output as lmc

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = SnowballStemmer('english')
def tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    stems = [stemmer.stem(x) for x in tokens]
    return stems

try:
    base_dir = '/home/ubuntu/localtype_site/localtype/data/'
    tfidf_vectorizer = dill.load(open(base_dir+'tfidf_vectorizer_nlpimport.m', 'rb'))
    c = dill.load(open(base_dir+"trained_pipeline_nlpimport.m", 'rb'))
except:
    print('first import failed, trying secondary import')
    base_dir = '/Users/nknezek/Documents/Insight_local/localtype_site/localtype/data/'
    tfidf_vectorizer = dill.load(open(base_dir + 'tfidf_vectorizer_nlpimport.m', 'rb'))
    c = dill.load(open(base_dir + "trained_pipeline_nlpimport.m", 'rb'))

# Load the text-analysis model

statetowns = ['Anchorage, AK', 'Berkeley, CA', 'Denton, TX']
explainer = LimeTextExplainer(class_names=statetowns)


def make_dropdown(towns, selected = 0):
    dropdown_html = "<select id=\"input_city\" name=\"input_city\">\n"
    for i,town in enumerate(towns):
        if i==selected:
            dropdown_html += "<option selected value=\"{}\">{}</option>\n".format(i,town)
        else:
            dropdown_html += "<option value=\"{}\">{}</option>\n".format(i, town)
    dropdown_html += "</select>\n"
    return dropdown_html

def compute_all_things(input_city, input_text):
    exp = explainer.explain_instance(input_text, c.predict_proba, num_features=6, labels=[int(input_city)], num_samples=500)
    syndict = syn.suggest_synonyms(c,tokenizer,input_text,int(input_city))
    synhtml = syn.html_suggested_synonyms(syndict)
    color_text = lmc.color_words(exp)
    # color_text = "<p>color text</p>"
    score = lmc.colored_score(exp,int(input_city))
    # top_cities = lmc.plot_cityscores(exp,int(input_city))
    top_cities = lmc.list_cities(exp,int(input_city))
    # top_cities = "<p>top cities</p>"

    top_words = lmc.plot_top_words(exp)
    # top_words = "<p>top words</p>"
    return score, color_text, top_words, top_cities, synhtml

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",)

@app.route('/output')
def text_output():
    # pull input text and city from input field and store it
    input_city = request.args.get('input_city')
    input_text = request.args.get('input_text')

    dropdown_html = make_dropdown(statetowns,int(input_city))
    if input_text == '':
        input_text = "you didn't enter any text! So instead you get to see this easter-egg! Aren't you lucky?"

    try:
        score, color_text, top_words, top_cities, synhtml = compute_all_things(input_city, input_text)
        return render_template("output.html", score=score, input_text=input_text, dropdown_html=dropdown_html, color_text=color_text, top_words=top_words, top_cities=top_cities, synonyms=synhtml)
    except:
        return render_template("error.html", dropdown_html=dropdown_html, input_text=input_text, input_city=input_city)

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/example1')
def example1():
    input_city = "1"
    input_text = "Try Nick's, an honest local business with good coffee and donuts!"

    dropdown_html = make_dropdown(statetowns,int(input_city))

    try:
        score, color_text, top_words, top_cities, synhtml = compute_all_things(input_city, input_text)
        return render_template("output.html", score=score, input_text=input_text, dropdown_html=dropdown_html, color_text=color_text, top_words=top_words, top_cities=top_cities, synonyms=synhtml)
    except:
        return render_template("error.html", dropdown_html=dropdown_html, input_text=input_text, input_city=input_city)


@app.route('/example2')
def example2():
    input_city = "1"
    input_text = "Try Nick's, a conscientious co-op with artisanal coffee and organic pastries!"

    dropdown_html = make_dropdown(statetowns,int(input_city))

    try:
        score, color_text, top_words, top_cities, synhtml = compute_all_things(input_city, input_text)
        return render_template("output.html", score=score, input_text=input_text, dropdown_html=dropdown_html, color_text=color_text, top_words=top_words, top_cities=top_cities, synonyms=synhtml)
    except:
        return render_template("error.html", dropdown_html=dropdown_html, input_text=input_text, input_city=input_city)


