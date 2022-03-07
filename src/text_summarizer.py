import string, re, os, math, json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from preprocess import open_file, split_text, tokenize
from topic_extraction import word_frequency, tf, tf_idf, idf

#load the word2vec pretrained model 
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

#makes centroid embeddings from word2vec vectors
def make_vec(words):
    embedding = [1]
    for x in range(len(words)):
        if words[x][1] in wv:
            embedding += wv[words[x][1]]
    return embedding

#makes the sentence vectors from word2vec embeddings
def sent_vec(sentences):
    embedding = list()
    #for each sentence
    for x in range(len(sentences)):
        temp = [1]
        #if the word is in word2vec, add to the embedding
        for word in sentences[x]:
            if word in wv:
                temp += wv[word]
        embedding.append(temp)
    return embedding
	
#get the cosine similarity of each sentence in a document and the centroid for each document
def cos_sim(centroid,corpus):
    cos_sim_sents = []
    #for each doucment
    for x in range(len(corpus)):
        #arrays should be np
        centroid_vec = np.array([centroid[x]])
        sentences = []
        #for each sentence in the document
        for y in range(len(corpus[x])):
            if len(corpus[x][y]) != 1:
                sentence = np.array([corpus[x][y]])
                #calculate the cosine similarity of the sentence and the centroid
                sentences.append((cosine_similarity(centroid_vec,sentence).tolist(),y))
            #sort sentences from high to low
            sentences.sort(reverse=True)
        cos_sim_sents.append(sentences)

    return cos_sim_sents
	
	
directory = '../data/'
files = os.listdir(directory)
stories = []
for file in files:
    filename = directory + '/' + file
    text = open_file(filename)
    stories.append(split_text(text))
    
#these hold the preprocessed and unprocessed summaries
corpus_p = list()
corpus_u = list()
temp_stories = list()
corpus_sentences = list()

#for each document, preprocess the document by tokenizing the sentences and words and removing stop words
for x in range(len(stories)):
    temp_processed, temp_unprocessed, sentences = tokenize(stories[x][0]['story'])
    corpus_p.append(temp_processed)
    corpus_u.append(temp_unprocessed)
    corpus_sentences.append(sentences)

#run the word frequency algorithm
wf_topic = list()
#get topic words for each document
for x in range(len(corpus_p)):
    wf_topic.append(word_frequency(corpus_p[x]))
        
#create the centroid embeddings for each article
wv_centroid = list()
for x in range(len(wf_topic)):
    wv_centroid.append(make_vec(wf_topic[x]))
    
#create vectors for every sentence in each article
wv_sentences = list()
for x in range(len(corpus_p)):
    wv_sentences.append(sent_vec(corpus_p[x]))
    
#calculate the cosine similarity between the setneces and the centroids
wf_cosine = cos_sim(wv_centroid,wv_sentences)



#run the tfidf algorithm
tf_voc = list()
#get the idf score for each document
idf_score = idf(corpus_p)
#get topic words for each document
tf_idf_topic = list()
for x in range(len(corpus_p)):
    tf_voc.append(tf(corpus_p[x]))
    tf_idf_topic.append(tf_idf(tf_voc[x],idf_score))
#get the centroid and sentence embeddings for tf-idf
tf_idf_centroid = list()
for x in range(len(tf_idf_topic)):
    tf_idf_centroid.append(make_vec(tf_idf_topic[x]))
tf_idf_sentences = list()
for x in range(len(corpus_p)):
    tf_idf_sentences.append(sent_vec(corpus_p[x]))
#calculate the cosine similarity of the centroid and the setnences
tf_idf_cosine = cos_sim(tf_idf_centroid,tf_idf_sentences)

#create summaries
sum_dict = {}
for x in range(len(stories)):
    tempdict = {}

    #this tells us how many sentences to extract from each document, it needs to be the same number of
    #sentences in the highlights because we are using ROUGE to compare them.
    cur = len(stories[x][0]['highlights'])
    
    tempstr = ""
    #add the reference summary to the json file
    for y in range(1,cur):
        tempstr += stories[x][0]['highlights'][y].strip('\n') + ". "
    #store the reference summary in the temporary dictionary
    tempdict['reference'] = tempstr
    
    #add the word2vec summary to the json file
    tempstr = ""
    for y in range(cur-1):
        #the summary is created by using the index of the highest scoring sentences
        #to get the correct sentences from the unprocessed article
        tempstr += TreebankWordDetokenizer().detokenize(corpus_u[x][wf_cosine[x][y][1]]) + " "
    #store the word2vec summary in the temporary dictionary
    tempdict['wf'] = tempstr
    
    
    #add the elmo summary to the json file
    tempstr = ""
    for y in range(cur-1):
        #the summary is created by using the index of the highest scoring sentences
        #to get the correct sentences from the unprocessed article
        tempstr += TreebankWordDetokenizer().detokenize(corpus_u[x][tf_idf_cosine[x][y][1]]) + " "
    #store the elmo summary in the temporary dictionary
    tempdict['tf_idf'] = tempstr
    
    #store the temporary dictionary in the summary dictionary which will contain all summaries
    sum_dict[str(x)] = tempdict
    
#write the summary dictionary to the json file
with open("summaries.json", "w") as outfile:
    json.dump(sum_dict, outfile,indent=4)