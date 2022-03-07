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

#extract topic words using word frequency
def word_frequency(text):
    vocabulary = {}
    total_word_length = 0
    for x in text:
        for y in x:
            total_word_length += 1
            if y in vocabulary:
                vocabulary[y] += 1
            else:
                vocabulary[y] = 1
    #add only the top 10% of words to the list
    highest = [(vocabulary[key], key) for key in vocabulary]
    highest.sort()
    highest.reverse()
    total = len(highest)
    top = total * 0.1
    topic = list()
    for x in range(int(top)):
        topic.append(highest[x])
        #print(highest[x])

    return topic
	
#caculate tf score of a document
def tf(text):
    vocabulary = {}
    total_word_length = 0
    for x in text:
        for y in x:
            total_word_length += 1
            if y in vocabulary:
                vocabulary[y] += 1
            else:
                vocabulary[y] = 1
    vocabulary.update((x,y/int(total_word_length)) for x, y in vocabulary.items())
    return vocabulary
	
#counts the number of times a word appears in all documents
def check_doc(word,docs):
    count = 1
    for x in docs:
        if word in x:
            count += 1
    return count

#calculate the idf score of all documents
def idf(docs):
    idf_score = {}

    temp_docs = list()
    for doc in docs:
        temp = list()
        for sent in doc:
            for word in sent:
                temp.append(word)
        temp_docs.append(temp)
    for doc in temp_docs:
        for word in doc:
            idf_score[word] = check_doc(word,temp_docs)
    idf_score.update((x, math.log(len(docs)/y)) for x, y in idf_score.items())
    
    return idf_score
	
#calculate the tf_idf score of a document
def tf_idf(tf_score, idf_score):
    tf_idf_score = {key: tf_score[key] * idf_score[key] for key in tf_score.keys()}
    highest = [(tf_idf_score[key], key) for key in tf_idf_score]
    highest.sort()
    highest.reverse()
    total = len(highest)
    top = total * 0.1
    topic = list()
    for x in range(int(top)):
        topic.append(highest[x])
    return topic