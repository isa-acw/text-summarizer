import string, re, os, math, json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from rouge import Rouge


with open('summaries.json') as json_file:
    data = json.load(json_file)
    
references = []
wf_summaries = []
tf_idf_summaries = []
for x in data:
    references.append(data[x]['reference'].translate(str.maketrans("","",string.punctuation)))
    wf_summaries.append(data[x]['wf'].translate(str.maketrans("","",string.punctuation)))
    tf_idf_summaries.append(data[x]['tf_idf'].translate(str.maketrans("","",string.punctuation)))
	
#make a rouge class object
rouge = Rouge()


#these will hold the rouge scores for the word2vec and elmo models
wf_rouge1 = [0,0,0]
wf_rouge2 = [0,0,0]
wf_rougeL = [0,0,0]
tf_idf_rouge1 = [0,0,0]
tf_idf_rouge2 = [0,0,0]
tf_idf_rougeL = [0,0,0]

#sum the rouge scores for word2vec and elmo summaries
for x in range(len(references)):
    #rouge scores for word2vec
    scores = rouge.get_scores(references[x],wf_summaries[x])
    wf_rouge1[0] += scores[0]['rouge-1']['r']
    wf_rouge1[1] += scores[0]['rouge-1']['p']
    wf_rouge1[2] += scores[0]['rouge-1']['f']

    wf_rouge2[0] += scores[0]['rouge-2']['r']
    wf_rouge2[1] += scores[0]['rouge-2']['p']
    wf_rouge2[2] += scores[0]['rouge-2']['f']

    wf_rougeL[0] += scores[0]['rouge-l']['r']
    wf_rougeL[1] += scores[0]['rouge-l']['p']
    wf_rougeL[2] += scores[0]['rouge-l']['f']
    
    #rouge scores for elmo
    scores = rouge.get_scores(references[x],tf_idf_summaries[x])
    tf_idf_rouge1[0] += scores[0]['rouge-1']['r']
    tf_idf_rouge1[1] += scores[0]['rouge-1']['p']
    tf_idf_rouge1[2] += scores[0]['rouge-1']['f']

    tf_idf_rouge2[0] += scores[0]['rouge-2']['r']
    tf_idf_rouge2[1] += scores[0]['rouge-2']['p']
    tf_idf_rouge2[2] += scores[0]['rouge-2']['f']

    tf_idf_rougeL[0] += scores[0]['rouge-l']['r']
    tf_idf_rougeL[1] += scores[0]['rouge-l']['p']
    tf_idf_rougeL[2] += scores[0]['rouge-l']['f']
	
	
#calculate the average of the rouge scores for word2vec and elmo
count = len(references)
for x in range(3):
    wf_rouge1[x] = wf_rouge1[x]/count
    wf_rouge2[x] = wf_rouge2[x]/count
    wf_rougeL[x] = wf_rougeL[x]/count

    tf_idf_rouge1[x] = tf_idf_rouge1[x]/count
    tf_idf_rouge2[x] = tf_idf_rouge2[x]/count
    tf_idf_rougeL[x] = tf_idf_rougeL[x]/count
	
#print results
print('\t\t\trecall\t\t     precision\t\t  f1')
print('ROUGE-1 wf:\t', wf_rouge1)
print('ROUGE-1 tf_idf:\t', tf_idf_rouge1)
print()
print('ROUGE-2 wf:\t',wf_rouge2)
print('ROUGE-2 tf_idf:\t', tf_idf_rouge2)
print()
print('ROUGE-L wf:\t',wf_rougeL)
print('ROUGE-L tf_idf:\t',tf_idf_rougeL)