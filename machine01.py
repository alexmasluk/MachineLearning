#!/usr/bin/python

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin


def load_data():
    '''Experiments 1 and 4
    Load olympic data
    Returns dict
    '''
    return sio.loadmat('olympics.mat')

def year_time_graph(olympics, dataset, fit_line=0):
    '''Experiment 2
    Reproduce fig 1.1 with male100 data
    '''
    data = olympics[dataset]
    year,time = data[:,0],data[:,1]
    plt.plot(year,time, 'bo')
    if fit_line == 1 or fit_line == 2:
        fit_model = order1_linear_regression(olympics, dataset)
        plt.plot([1895,2010], [fit_model.predict(1895), fit_model.predict(2010)])
    if dataset == 'male100':
        plt.axis([1880,2020,9.5,12])
    if dataset == 'female400':
        plt.axis([1960,2010, 47, 53])

    plt.xlabel('Year')
    plt.ylabel('Time (seconds)')
    plt.show()

def order1_linear_regression(olympics, keyname):
    '''Perform first-order linear regression
    Params: the olympics.dat datasets, the name of dataset on which to do LR
    Returns the LR fit object
    '''
    lr = lin.LinearRegression()
    dat = olympics[keyname]
    x = dat[:,0].reshape(-1,1)
    y = dat[:,1]
    return lr.fit(x,y)


def test_regression_model(olympics, test_type=1):
    '''Experiments 3 and 5
    Evaluate a linear regression model and display results
    '''
    if test_type == 1:
        model = order1_linear_regression(olympics, 'male100')
        #experiment 3: display coefficients and predictions
        #compare results with Section 1.2
        Y1, Y2 = 9.595, 9.541
        x1, x2 = 2012, 2016
        print("coefficients:")
        print("f(x ; w) = {} + {}x".format( str(model.intercept_), str(model.coef_[0])))
        y1 = round(float(model.predict(x1)), 3)
        y2 = round(float(model.predict(x2)), 3)
        print("predictions:")
        print("f({}) = {}".format(x1,y1))
        print("f({}) = {}".format(x2,y2))
        print("comparison ( expected == actual? ):")
        print("{} == {}? {}".format(Y1, y1, Y1 == y1))
        print("{} == {}? {}".format(Y2, y2, Y2 == y2))



def main():
    olympics = load_data()
    year_time_graph(olympics, 'male100')
    test_regression_model(olympics, 1)
    year_time_graph(olympics, 'male100', 1)
    year_time_graph(olympics, 'female400', 1)



if __name__ == '__main__':
    main()
