#!/usr/bin/python

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
import numpy as np

classes = ['EAP', 'HPL', 'MWS']

def load_data():
    with open('train.txt') as f:
        trn = f.readlines()

    with open('test.txt') as f:
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
    vects = []
    Xs = []
    for x in range(1,2):
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range = (1,2))

        X = vectorizer.fit_transform(dat)
        vects.append(vectorizer)
        Xs.append(X)
    return Xs, vects

def extract_test_feat(vect, dat):
    X = vect.transform(dat)
    return X

def reduce_features(X_train, y_train, X_val, num_feat):
    ch2 = SelectKBest(chi2, k=num_feat)
    X_train = ch2.fit_transform(X_train, y_train)
    X_val = ch2.transform(X_val)
    return X_train, X_val

def pass_agg(X_train, y_train, X_val, y_val):
    clf = PassiveAggressiveClassifier(n_iter=50)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    p_pred = clf.decision_function(X_val)

    score = metrics.accuracy_score(y_val, pred)
    print("accuracy: {}".format(score))
    return (clf, score)


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
    return (clf, score)


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
    X_trains, vectorizers = train_vect(X_train)

    most_accurate = 0
    best_clf = None
    best_vect = None
    for i, vect in enumerate(vectorizers):
        # use vectorizer to get vectorized version of validation dat
        X_v = extract_test_feat(vect, X_val)

        # grab feature names
        feature_names = vect.get_feature_names()

        # process them for some reason
        f_n = np.asarray(feature_names)

        # reduce features maybe we'll see
        num_feat = len(vect.get_feature_names())
        X_tr, X_v = reduce_features(X_trains[i], y_train, X_v, num_feat)

        # train a classifier
        #clf = ridge(X_tr, y_train, X_v, y_val)
        clf = pass_agg(X_tr, y_train, X_v, y_val)
        if clf[1] > most_accurate:
            most_accurate = clf[1]
            best_vect = vect
            best_clf = clf[0]


    # classify the test data for submission
    final_test_ids = [x[0] for x in final_test[1:]]
    final_test_data = [x[1] for x in final_test[1:]]
    final_test_data = extract_test_feat(best_vect, final_test_data)

    probs = evaluate(best_clf, final_test_data)
    write(probs, final_test_ids)


if __name__ == '__main__':
    main()

