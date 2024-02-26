import os
import random
import string
from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

BASE_DIR = ''
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']

def create_data_set():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        for label in LABELS:
            dir = '%s/%s' % (BASE_DIR, label)
            for filename in os.listdir(dir):
                fullfilename = '%s/%s' % (dir, filename)
                print(fullfilename)
                with open(fullfilename, 'rb') as file:
                    text = file.read().decode(errors= 'replace').replace('\n', '')
                    outfile.write('%\t%s\t%s\n' % (label, filename, text))


if __name__ == '__main__':
    create_data_set()
