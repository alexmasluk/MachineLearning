#!/usr/bin/python
#import scipy.io as sio

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def load_data():
    '''Load and pre-process the data
    '''
    with open('dresses1.csv') as f:
        data = f.readlines()


    # the dataset's first row is the column headers, ignoring ID column
    column_headers = data[0].strip().split(',')[1:]
    num_columns = len(column_headers)

    # dictionary needed to accumulate values for qual. vars
    category_lists = {}
    for heading in column_headers:
        category_lists[heading] = []

    # format the data as a list of lists
    dataset  = []
    response = []
    for datum in data[1:]:
        features = datum.strip().split(',')[1:]
        dataset.append(features[:-1])
        response.append(float(features[-1:][0]))

        # track list of qual. var values
        for x, feature in enumerate(features):
            category_lists[column_headers[x]].append(feature)


    # remove duplicate values
    for header in column_headers:
        category_lists[header] = list(set(category_lists[header]))

    # replace qual. vars with indices
    data2 = []
    c_feat = []
    for datum in dataset:
        new_feat = []
        for x, feat in enumerate(datum):
            header = column_headers[x]
            try:
                f_val = float(feat)
                new_feat.append(f_val)
            except ValueError:
                if x not in c_feat:
                    c_feat.append(x)
                index = category_lists[header].index(feat)
                new_feat.append(index)
        data2.append(new_feat)

    dataset = data2

    # use OneHotEncoder to replace categorical values with binary vectors
    enc = OneHotEncoder(categorical_features = c_feat)
    enc.fit(dataset)

    data2 = []
    for dat in dataset:
        arr = enc.transform([dat]).toarray()
        data2.append(arr)

    dataset = data2

    return dataset, response

def split_data(data, response):
    '''Experiment 1
    Use train_test_split to split in to training, test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
            data, response, test_size = 0.25)

    return (X_train, X_test), (y_train, y_test)

def reshape(d):
    data = np.array(d)
    data = np.array(data)
    nsamples, nx, ny = data.shape
    return data.reshape(nsamples, nx*ny)


def gaussian_nb(data, response):
    clf = GaussianNB()
    data = reshape(data)
    clf.fit(data, np.array(response).ravel())
    return clf

def test_gauss(clf, data, response):
    good, bad = 0,0
    for i, x in enumerate(data):
        pred = clf.predict(x)
        actual = response[i]
        if pred == actual:
            good += 1
        else:
            bad += 1
    print("good {} bad {}".format(good, bad))

    scores = clf.predict(reshape(data))
    fpr, tpr, thresholds = roc_curve(response, scores, pos_label=1)
    roc_score = roc_auc_score(response, scores)
    print("FPR",fpr)
    print("TPR",tpr)
    print("THR",thresholds)
    print("AUC",roc_score)

def knn(data, response):
    data = reshape(data)
    print(len(data))
    n = int(np.sqrt(len(data)))
    k_classifiers = []
    for x in range(5, n):
        k = KNeighborsClassifier(n_neighbors=x)
        k.fit(data, response)
        k_classifiers.append(k)

    return k_classifiers


def main():
    # setup
    data, response = load_data()

    # experiment 1
    print("Experiment 1: Split into test and training")
    data, response = split_data(data, response)
    print("Experiment 1 complete\n")

    # experiment 2
    print("Experiment 2: Fit GaussianNB classifier to data")
    clf = gaussian_nb(data[0], response[0])
    print("Experiment 2 complete\n")

    # experiment 3
    print("Experiment 3: Use the classifier to predict labels, and eval performance")
    test_gauss(clf, data[1], response[1])
    print("Experiment 3 complete\n")


    # experiment 4
    print("Experiment 4: Fit KNN classifier to data, evaluate")
    klassifiers = knn(data[0], response[0])
    print(len(klassifiers))

    print("Experiment 4 complete\n")



if __name__ == '__main__':
    main()

