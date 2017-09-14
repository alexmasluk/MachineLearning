#!/usr/bin/python

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures as poly
from sklearn.pipeline import make_pipeline

def load_data():
    '''Experiment 1
    Load olympic data
    Return dict
    '''
    return sio.loadmat('olympics.mat')

def year_time_graph(olympics, dataset, fit_line=0):
    '''Experiments 2 and 4
    Reproduce figures 1.1 and 1.5 from textbook
    '''
    data = olympics[dataset]
    year,time = data[:,0],data[:,1]
    plt.plot(year,time, 'bo')
    if fit_line == 1:
        fit_model = order1_linear_regression(olympics, dataset)
        plt.plot([1895,2010], [fit_model.predict(1895), fit_model.predict(2010)])
    if dataset == 'male100':
        plt.axis([1880,2020,9.5,12])

    plt.xlabel('Year')
    plt.ylabel('Time (seconds)')
    print('Please close the graph to continue...')
    plt.show()

def order1_linear_regression(olympics, dataset):
    '''Perform first-order linear regression
    Params: the olympics.dat datasets, the name of dataset on which to do LR
    Return the LR fit object
    '''
    lr = lin.LinearRegression()
    dat = olympics[dataset]
    x = dat[:,0].reshape(-1,1)
    y = dat[:,1]
    return lr.fit(x,y)

def linear_regression_order_n(olympics, dataset, degree=1):
    '''Perform n-order linear regression
    Params: the olympics datasets, name of dataset on which to do LR, polynomial degree
    Returns pipeline object
    '''
    lr = lin.LinearRegression()
    dat = olympics[dataset]
    x = dat[:,0].reshape(-1,1)
    y = dat[:,1]
    model = make_pipeline(poly(degree), lr)
    return model.fit(x,y)




def test_regression_model(olympics, test_type=1, degree=3, compare=None):
    '''Experiments 3, 5, 6, 7
    Evaluate linear regression model(s) and display results
    Params: Olympics datasets, test to perform, polynomial degree, previous value for comparison
    Return value for future comparison
    '''

    # Occasionally we want to remember a value from our tests
    # This gets returned
    value = None
    if test_type == 1:
        model = order1_linear_regression(olympics, 'male100')
        # Experiment 3: display coefficients and predictions
        # Compare results with Section 1.2
        Y1, Y2 = 9.595, 9.541
        x1, x2 = 2012, 2016
        print("Coefficients:")
        print("f(x ; w) = {} + {}x".format( str(model.intercept_), str(model.coef_[0])))
        y1 = round(float(model.predict(x1)), 3)
        y2 = round(float(model.predict(x2)), 3)
        print("Predictions:")
        print("f({}) = {}".format(x1,y1))
        print("f({}) = {}".format(x2,y2))
        print("Comparison ( expected == actual? ):      #expected values from textbook")
        print("{} == {}? {}".format(Y1, y1, Y1 == y1))
        print("{} == {}? {}".format(Y2, y2, Y2 == y2))


    if test_type == 2:
        # Experiment 5: fit a line to female400
        # Compare error with error from male100

        # X vectors
        m_x_vector = olympics['male100'][:,0].reshape(-1,1)
        f_x_vector = olympics['female400'][:,0].reshape(-1,1)

        # Actual values
        m_actual = olympics['male100'][:,1]
        f_actual = olympics['female400'][:,1]

        # Linear regression models
        model_m = order1_linear_regression(olympics, 'male100')
        model_f = order1_linear_regression(olympics, 'female400')

        # Predicted values
        m_pred = model_m.predict(m_x_vector)
        f_pred = model_f.predict(f_x_vector)

        # Mean squared error for both sets
        m_error = mse(m_actual, m_pred)
        f_error = mse(f_actual, f_pred)

        # Display results
        print("male100 mse  : {}".format(m_error))
        print("female400 mse: {}".format(f_error))
        print("female_mse - male_mse = {}".format(abs(f_error - m_error)))
        value = f_error

    if test_type == 3:
        # Experiment 6
        # Fit an N order polynomial to female400. Does the error improve?

        # Get the model
        poly_model = linear_regression_order_n(olympics, 'female400', degree)

        # X vector and actual values
        x_vector = olympics['female400'][:,0].reshape(-1,1)
        actual = olympics['female400'][:,1]

        # Prediction set and mean squared error
        p_pred = poly_model.predict(x_vector)
        poly_error = mse(actual, p_pred)

        # Display results
        print("{}-degree error: {}".format(degree, poly_error))
        print("Previous error:  {}".format(compare))
        print("Current error < previous error? {}".format(poly_error < compare))
        print("Diff = {}".format(abs(poly_error - compare)))
        value = poly_error



    raw_input("Please hit enter to continue...")
    return value



def main():
    print("Experiment 1: Load the data")
    olympics = load_data()
    print("Data loaded: type(olympics) = {}".format(type(olympics)))
    print('')

    print("Experiment 2: Reproduce figure 1.1")
    year_time_graph(olympics, 'male100')
    print('')

    print("Experiment 3: Get linear regression model for male100")
    print("              List coefficients and compare predictions")
    test_regression_model(olympics, 1)
    print('')

    print("Experiment 4: Reproduce figure 1.5")
    year_time_graph(olympics, 'male100', 1)
    print('')

    print("Experiment 5: Get linear regression model for female400")
    print("              Compare error with model for male100")
    error = test_regression_model(olympics, 2)
    print('')

    print("Experiment 6: Fit a 3rd degree polynomial to female400")
    print("              Does the error improve?")
    error = test_regression_model(olympics, 3, degree=3, compare=error)
    print('')


    print("Experiment 6: Fit a 5th degree polynomial to female400")
    print("              Does the error improve?")
    test_regression_model(olympics, 3, degree=5, compare=error)
    print('')

if __name__ == '__main__':
    main()
