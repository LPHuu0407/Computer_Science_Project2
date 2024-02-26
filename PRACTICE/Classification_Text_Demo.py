import os
import random
import string
from nltk import word_tokenize
from collections import defaultdict
from nltk import FeatDict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle


def create_data_set():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        print()


if __name__ == '__main__':
    create_data_set()
