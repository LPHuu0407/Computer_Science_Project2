import tkinter as tk
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
from Create_Dataset import create_data_set
# loại bỏ các stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')
# Đọc bộ dữ liệu đã được tạo, là bước đầu tiên và quan trọng để mô hình hoạt động
def setup_docs():
    docs = [] # (label, text)
    with open('D:\MYLEARNING\THE_JOURNEY_IV\COMPUTER_SCIENCE_PROJECT_2\PRACTICE\data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split('\t')
            doc = ( parts[0], parts[2].strip() )
            docs.append(doc)
        return docs
# Làm sạch văn bản, loại bỏ các ký tự, khoảng trắng dư thừa
def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lower case
    text = text.lower()
    return text
# Tách từ trong văn bản và loại bỏ stopwords
def get_tokens(text):
    #get individual words
    tokens = word_tokenize(text)
    # remove common words that are useless
    tokens = [t for t in tokens if not t in stop_words]
    return tokens
# Hàm này thực hiện các bước cơ bản, làm sạch văn bản, tách từ
def print_frequency_dist(docs):
    tokens = defaultdict(list)
    # lets make a giant list of all the words for each category
    # hãy tạo một danh sách khổng lồ gồm tất cả các từ cho mỗi danh mục
    for doc in docs:
        doc_label = doc[0]
        #doc_text = doc[1] sau khi đã tìn ra các từ xuất hiện nhiều nhất, đến bước clean text #1
        doc_text = clean_text(doc[1]) # clean text, xóa bỏ các dấu câu #2
        #doc_tokens = word_tokenize(doc_text) #3
        doc_tokens =get_tokens(doc_text) #4
        tokens[doc_label].extend(doc_tokens)

    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))
def get_splits(docs):
    # scramb docs
    random.shuffle(docs)
    X_train = []# training documents
    y_train = []# corresponding training labels
    X_test = []# test documents
    y_test = []# corresponding test labels
    pivot = int(.80 * len(docs))
    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])
    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])
    return X_train, X_test, y_train, y_test
def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))
def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs)
    # the object that turns text into vector
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=3, analyzer='word')
    # create doc-term matrix
    dtm = vectorizer.fit_transform(X_train)
    # train Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)
    evaluate_classifier("Naive Bayes\tTRAI\t", naive_bayes_classifier, vectorizer, X_train, y_train)
    evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes_classifier, vectorizer, X_test, y_test)
    # store the classifier 
    clf_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))
    # also store the vectorizer so we transform new data
    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))
def classify(text):
    # load classifier
    clf_filename = 'D:\\MYLEARNING\\THE_JOURNEY_IV\COMPUTER_SCIENCE_PROJECT_2\\naive_bayes_classifier.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))
    # vectorize the new text
    vec_filename = 'D:\\MYLEARNING\\THE_JOURNEY_IV\\COMPUTER_SCIENCE_PROJECT_2\\count_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))
    pred = nb_clf.predict(vectorizer.transform([text]))
    print(pred[0])
if __name__ == '__main__':
    #create_data_set()
    #docs = setup_docs()
    #print_frequency_dist(docs)
    #train_classifier(docs)
    new_doc = """  After 2021’s first entry fared similarly well – and picked up a total of six Academy Awards – the zeitgeist is inevitably comparing director Denis Villeneuve’s sprawling world of “Dune” to those of “Star Wars” and “Lord of the Rings.” With that said, there’s no time like the present to use Arrakis, the desert planet at the heart of “Dune,” as the jumping-off point to explore other amazing sequels in science-fiction, a genre that’s luckily been fertile ground for various second chapters in cinema. """
    classify(new_doc)
    print("Done")