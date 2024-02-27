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

# BASE_DIR = 'D:\\MYLEARNING\\THE_JOURNEY_IV\\COMPUTER_SCIENCE_PROJECT_2\\PRACTICE\\bbc'
# LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']


# def create_data_set():
#     with open('data.txt', 'w', encoding='utf8') as outfile:
#         for label in LABELS:
#             dir = '%s/%s' % (BASE_DIR, label)
#             for filename in os.listdir(dir):
#                 fullfilename = '%s/%s' % (dir, filename)
#                 print(fullfilename)
#                 with open(fullfilename, 'rb') as file:
#                     text = file.read().decode(errors= 'replace').replace('\n', '')
#                     outfile.write('%\t%s\t%s\n' % (label, filename, text))

def setup_docs():
    docs = [] # (label, text)
    with open('D:\MYLEARNING\THE_JOURNEY_IV\COMPUTER_SCIENCE_PROJECT_2\PRACTICE\data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split('\t')
            doc = ( parts[0], parts[2].strip() )
            docs.append(doc)
        return docs

def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lower case
    text = text.lower()
    return text

def print_frequency_dist(docs):
    tokens = defaultdict(list)
    # lets make a giant list of all the words for each category
    # hãy tạo một danh sách khổng lồ gồm tất cả các từ cho mỗi danh mục
    for doc in docs:
        doc_label = doc[0]
        doc_text = doc[1]
        doc_tokens = word_tokenize(doc_text)
        tokens[doc_label].extend(doc_tokens)
    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))


if __name__ == '__main__':
    #create_data_set()
    docs = setup_docs()
    print_frequency_dist(docs)
    #train_classifier(docs)
    # new_doc = "Google showed..."
    #classify(new_doc)
    print("Done")
