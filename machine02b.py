#!/usr/bin/python
#import scipy.io as sio

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, zero_one_loss, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from itertools import permutations
import numpy as np

def load_data():
    '''Load and pre-process the data
    '''
    print("loading data")
    with open('human.csv') as f:
        data = f.readlines()


    # the dataset's first row is the column headers, ignoring ID column
    column_headers = data[0].strip().split(',')[1:]
    num_columns = len(column_headers)

    # format the data as a list of lists keep labels separate
    dataset  = []
    response = []
    x = 0

    for datum in data:
        #print(x)
        features = datum.strip().split(',')[1:]
        dataset.append(features[:-1])
        response.append(int(features[-1:][0]) - 1)
        x += 1
    return np.array(dataset).astype(np.float), response

def split_data(data, response):
    '''Experiment 1
    Use train_test_split to split in to training, test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
            data, response, test_size = 0.25)

    return (X_train, X_test), (y_train, y_test)


def gaussian_nb(data, response):
    '''Experiment 2
    Fit gaussian naive bayes classifier to data
    '''
    clf = GaussianNB()
    clf.fit(data, np.array(response).ravel())
    return clf


def test_nb(clf, data, response):
    '''Experiment 3
    Test naive bayes classifier
    Display confusion matrix
    Return average loss for comparison with other classifiers
    '''
    predict = clf.predict(data)
    c = confusion_matrix(response, predict)
    print("CONFUSION MATRIX")
    print(c)
    FP = c.sum(axis=0) - np.diag(c)  
    FN = c.sum(axis=1) - np.diag(c)
    TP = np.diag(c)
    TN = c.sum() - (FP + FN + TP)
    
    performance = np.mean((TP+TN)/(TP+FP+FN+TN))

    print("performance: {}".format(performance))
    return (clf, performance)


def knn(data, response):
    '''Experiment 4
    Fit K Neighbors classifiers to data
    NOTE we did not use range k=5 to sqrt(N) due to large N and high dimensionality
    Please see separate text file for output of testing all values of K
    '''
    n = int(np.sqrt(len(data))) + 1
    print(len(data), n)
    k_classifiers = []
    for x in range(3, 6):
        print('fitting knn with k = {}'.format(x))
        k = KNeighborsClassifier(n_neighbors=x)
        k.fit(data, response)
        k_classifiers.append(k)

    return k_classifiers

def test_classifiers(classifiers, data, response):
    '''Experiment 5, 7
    Test classifier performance with zero_one_loss
    '''

    scores = []
    for c in classifiers:
        print("testing classifier: {}".format(c))
        pred = c.predict(data)
        scores.append( (c, zero_one_loss(response, pred)) )

    for score in scores:
        print(score)

    return scores

def svm(data, response):
    '''Experiment 6
    Fit 3 svm models to data
    Kernel = linear, kernel = poly with deg 3, kernel = poly with degree 5
    '''
    max_it = -1
    v=False
    clf_l, clf_p3, clf_p5 = SVC(kernel = "linear", verbose=v, max_iter=max_it), SVC(
            kernel = "poly", verbose=v, max_iter=max_it), SVC(
                    kernel = "poly", degree=5, verbose=v, max_iter=max_it)
    print("fitting with kernel=linear")
    clf_l.fit(data, response)

    print("fitting with kernel=poly, degree 3")
    clf_p3.fit(data, response)

    print("fitting with kernel=poly, degree 5")
    clf_p5.fit(data, response)
    return [clf_l, clf_p3, clf_p5]

def kmeans(data):
    '''Experiment 8
    use k means clustering to cluster the dataset
    return the fit model so we can determine if clustering corresponds to class labels
    '''
    km = KMeans(n_clusters = 6)
    km.fit(data)
    return km

def test_km(pred, actual):
    '''Experiment 8
    Our k-means didn't necessarily label the classes in the same order, so
    we'll score all possible class order permutations and use the best one to determine
    how accurately it found clusters that correspond to our actual classes
    '''
    mappings = list(permutations(['a','b','c','d','e','f']))
    l_to_d = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5}

    best_score = 1.0


    for mapping in mappings:
        d_to_l = {}
        for x, m in enumerate(mapping):
            d_to_l[x] = m
        remapped_pred = [l_to_d[d_to_l[p]] for p in pred]
        '''
        for p in pred:
            remapped_pred.append(l_to_d[d_to_l[p]])
            '''
        score = zero_one_loss(remapped_pred, actual)
        if score < best_score:
            best_score = score

    return best_score


def pick_best_classifier(scores):
    '''Choose the highest-performing classifier from a list
    Return the classifier and its score
    '''
    best = 1.0
    best_c = None
    for score in scores:
        if score[1] < best:
            best = score[1]
            best_c = score[0]

    return (best_c, best)

def main():
    # setup
    data, response = load_data()
    scores = []

    # experiment 1
    print("Experiment 1: Split into test and training")
    data, response = split_data(data, response)
    print("Experiment 1 complete\n")
    input("Press enter to continue . . .")

    # standardize for svm
    #  ** our dataset is already standardized **
    #data = standardize(data)

    # experiment 2
    print("Experiment 2: Fit GaussianNB classifier to data")
    clf = gaussian_nb(data[0], response[0])
    print("Experiment 2 complete\n")
    input("Press enter to continue . . .")

    # experiment 3
    print("Experiment 3: Use the classifier to predict labels, and eval performance")
    scores.append(test_nb(clf, data[1], response[1]))
    print("Experiment 3 complete\n")
    input("Press enter to continue . . .")


    # experiment 4
    print("Experiment 4: Fit KNN classifier to data, evaluate")
    k_classifiers = knn(data[0], response[0])
    print("Testing KNN classifiers..")
    scores.extend(test_classifiers(k_classifiers, data[1], response[1]))
    print("Experiment 4 complete\n")
    input("Press enter to continue . . .")

    # experiment 5
    print("Experiment 5: Which is better, our best KNN classifier or our Naive Bayes?")
    best = pick_best_classifier(scores)
    print(best[0], best[1])
    print("Experiment 5 complete\n")
    input("Press enter to continue . . .")

    # experiment 6
    print("Experiment 6: Fit SVM classifier to data, evaluate")
    svms = svm(data[0], response[0])
    scores.extend(test_classifiers(svms, data[1], response[1]))
    print("Experiment 6 complete\n")
    input("Press enter to continue . . .")

    # experiment7
    print("Experiment 7: Which is better, our best KNN classifier or our Naive Bayes?")
    best = pick_best_classifier(scores)
    print(best[0], best[1])
    print("Experiment 7 complete\n")
    input("Press enter to continue . . .")

    # experiment 8
    print("Experiment 8: Remove labels, use KMeans to cluster data")
    km = kmeans(data[0])
    score = test_km(list(km.labels_), response[0])
    print("{} loss for k means".format(score))
    print("Experiment 8 complete")
    input("Press enter to continue . . .")


if __name__ == '__main__':
    main()

