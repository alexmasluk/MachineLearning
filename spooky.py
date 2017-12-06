#!/usr/bin/python

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import numpy as np

classes = ['EAP', 'HPL', 'MWS']

def load_data():
    with open('train.txt', encoding='utf-8') as f:
        trn = f.readlines()

    with open('test.txt', , encoding='utf-8') as f:
        tst = f.readlines()

    train = [x.strip().split('\t') for x in trn]
    test  = [x.strip().split('\t') for x in tst]

    return train, test


def t_split(train):
    '''
    train_"validation"_split
    '''
    X = [x[1] for x in train[1:]]
    y = [y[2] for y in train[1:]]
    return train_test_split(X,y)

def train_vect(dat):
    vectorizer = TfidfVectorizer(sublinear_tf=True)

    X = vectorizer.fit_transform(dat)
    return X, vectorizer

def extract_test_feat(vect, dat):
    X = vect.transform(dat)
    return X

def reduce_features(train, test):
    pass


def ridge(X_train, y_train, X_val, y_val):
    clf = RidgeClassifier(solver="sag")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    p_pred = clf.decision_function(X_val)

    '''
    for x, p in enumerate(pred[:10]):
        s = "{} {}".format(p, (np.exp(p_pred[x]) / np.sum(np.exp(p_pred[x]))))
        print(s)
    '''

    score = metrics.accuracy_score(y_val, pred)
    print("accuracy: {}".format(score))
    return clf

def evaluate(clf, t_dat):
    pred = clf.decision_function(t_dat)
    probs = []
    for d in pred:
        probs.append(np.exp(d) / np.sum(np.exp(d)))

    return probs

def write(probs, final_test_ids):
    with open('submission.csv', 'w+') as f:
        f.write('id,EAP,HPL,MWS\n')
        for x, p in enumerate(probs):
            s = "{},{},{},{}\n".format(final_test_ids[x], p[0], p[1], p[2])
            f.write(s)


def main():
    # load the data from files; final_test used later
    train, final_test = load_data()

    # split the training data into training and "test" (validation) sets
    X_train, X_val, y_train, y_val = t_split(train)

    # get the vectorizer and the vectorized matrix based on training dat
    X_train, vectorizer = train_vect(X_train)

    # use vectorizer to get vectorized version of validation dat
    X_val = extract_test_feat(vectorizer, X_val)

    # grab feature names
    feature_names = vectorizer.get_feature_names()

    # process them for some reason
    f_n = np.asarray(feature_names)

    # reduce features maybe we'll see
    reduce_features(X_train, y_train)

    # train a classifier
    clf = ridge(X_train, y_train, X_val, y_val)

    # classify the test data for submission
    final_test_ids, final_test_data = [x[0] for x in final_test[1:]], [x[1] for x in final_test[1:]]

    final_test_data = extract_test_feat(vectorizer, final_test_data)

    probs = evaluate(clf, final_test_data)
    write(probs, final_test_ids)


if __name__ == '__main__':
    main()

