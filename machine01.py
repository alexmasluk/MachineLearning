#!/usr/bin/python

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt



def load_data():
    '''Experiment 1
    Load olympic data
    Returns dict
    '''
    return sio.loadmat('olympics.mat')

def year_time_graph(male100):
    '''Experiment 2
    Reproduce fig 1.1 with male100 data
    '''
    year,time = male100[:,0],male100[:,1]
    plt.plot(year,time, 'bo')
    plt.axis([1880,2020,9.5,12])
    plt.xlabel('Year')
    plt.ylabel('Time (seconds)')
    plt.show()


def main():
    olympics = load_data()
    year_time_graph(olympics['male100'])


if __name__ == '__main__':
    main()
