#!/usr/bin/python

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin


def load_data():
    '''Experiment 1
    Load olympic data
    Returns dict
    '''
    return sio.loadmat('olympics.mat')

def year_time_graph(olympics):
    '''Experiment 2
    Reproduce fig 1.1 with male100 data
    '''
    male100 = olympics['male100']
    year,time = male100[:,0],male100[:,1]
    plt.plot(year,time, 'bo')
    plt.axis([1880,2020,9.5,12])
    plt.xlabel('Year')
    plt.ylabel('Time (seconds)')
    plt.show()

def test_regression_model(model, test_type=1):
    '''Experiments 3 and 5
    Evaluate a linear regression model and display results
    '''
    if test_type == 1:
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


def main():
    olympics = load_data()
    year_time_graph(olympics)
    l_mod = order1_linear_regression(olympics, 'male100')
    test_regression_model(l_mod, 1)



if __name__ == '__main__':
    main()
