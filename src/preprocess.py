import string, re, math
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#opens the files
def open_file(filename):
    file = open(filename, encoding='utf-8')
    text = file.read()
    file.close()
    return text
	
#splits the text from the files into stories and highlights (reference summaries)
def split_text(text):
    stories = list()
    index = text.find('@highlight')
    doc, highlights = text[:index], text[index:].split('@highlight')
    stories.append({'story':doc,'highlights':highlights})

    return stories
	
#tokenizes the text to create sentence tokens, word tokens, and removes stop words
def tokenize(text):
    #we want to keep the processed and unprocessed text
    processed = list()
    unprocessed = list()
    #tokenize sentences
    sentences = sent_tokenize(text)
    for sentence in sentences:
        new_sent = list()
        #tokenize the words in the sentences
        unprocessed.append(word_tokenize(sentence))
        #remove punctuation
        tokens = word_tokenize(sentence.lower().translate(str.maketrans("","",string.punctuation)))
        #remove stop words from the sentences
        filtered_sent = [word for word in tokens if word not in stopwords.words('english')]
        for w in filtered_sent:
            new_sent.append(w)
            #new_sent.append(ps.stem(w))
            #print(w, " : ", ps.stem(w))
        processed.append(new_sent)

    #return the processed text, the original text, and the tokenized sentences in the text
    return processed, unprocessed, sentences