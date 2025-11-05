import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from collections import defaultdict, Counter

nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

def cleantext(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenizetext(text):
    return text.split()

def removestopwordstokens(tokens):
    return [t for t in tokens if t not in stop_words]

def stemtokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]

def preprocess(text):
    text = cleantext(text)
    tokens = tokenizetext(text)
    tokens = removestopwordstokens(tokens)
    tokens = stemtokens(tokens)
    return tokens
