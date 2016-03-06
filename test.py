#coding:utf8

import numpy as np, scipy
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from itertools import *
import collections
from multiprocessing import Pool
import random
from scipy.optimize import leastsq

class Test:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

if __name__ == '__main__':
    # arr = np.arange(9).reshape((3, 3))
    # print arr
    # np.random.shuffle(arr[:])
    # print arr

    arr = np.arange(9)
    print arr[0:255]
    print [1,2,3].append([4,5,6])
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print np.vstack((a,b))
    print a
    t = Test(name='hello', age=15)
    print t.name
    print dir(t)
    arr = np.array([[1,1],[2,2],[3,3]])
    print arr
    print arr[arr[:, 1] > 2]
    print zip(np.arange(3), np.arange(3))
    arr = np.array([1,2,3,4,5])
    arr2 = np.array(arr)
    arr[2] = 999
    print arr