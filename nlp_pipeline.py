from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = SnowballStemmer('english')

def tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    stems = [stemmer.stem(x) for x in tokens]
    return stems

def make_tfidf_vectorizer(vocab_stems, stop_stems):
    tfidf_vectorizer = TfidfVectorizer(vocabulary=list(vocab_stems), stop_words=stop_stems, tokenizer=tokenize)
    return tfidf_vectorizer

def make_pipeline(tfidf_vectorizer, classifier):
    c = Pipeline(steps=[('tfidf_vectorizer',tfidf_vectorizer), ('classifier',classifier)])
    return c
