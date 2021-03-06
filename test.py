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
import itertools

class Test:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

# def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n):
#         arr = reduce(lambda x,y: x+y, map(lambda j: [arr[j]] if arr[j-1] > arr[j] else [arr[j-1]], range(1, n-i)), []) + arr[n-i+1:]
#         print arr, len(arr)
#     return arr

# def bubble_sort(arry):
#     n = len(arry)                   #获得数组的长度
#     for i in range(n):
#         for j in range(1,n-i):
#             if  arry[j-1] > arry[j] :       #如果前者比后者大
#                 arry[j-1],arry[j] = arry[j],arry[j-1]      #则交换两者
#     return arry


def belong_point(args, point):
    # if possible, add one point to one of the surfaces

    A = np.array([args[0], args[1], 1.0, args[2]]).reshape(1, 4)
    X = np.array([point[0], point[1], point[2], 1.0]).reshape(4, 1)
    print A
    print X
    upper = np.dot(A, X)
    print "upper", upper
    print 'a', A[:, 0:3]
    lower = math.sqrt(np.dot(A[:, 0:3], (A[:, 0:3]).reshape(3,1)))
    print "lower", lower
    # lower = 1
    e = float(abs(upper / lower))
    return e

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
    alist = [54,26,93,17,77,31,44,55,20]  
    # print bubble_sort(alist)
    arr = [1,2,3,4,5]
    for x in arr:
        if x == 3:
            arr.remove(x)
    print arr
    arr = np.array([1,2,3,4,5])
    mask = np.array([True, False, False, False, True])
    print arr[mask]
    print np.array([]).shape
    arr = np.array([1,2,3,4,5])
    print arr[0:3]
    arr = np.array([3.0, 4.0])
    print np.linalg.norm(arr)
    arr = np.array([[1.0,1.0], [100.0, 100.0]])
    print np.linalg.norm(arr)
    arr = np.array([1,2,3,4,5])
    print arr[arr > 3]
    a = np.array([[1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 1, 0]])

    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]
    print unique_a

    print belong_point([1.47701851e-11, -1.18233201e-11, -2.99883268e+01], [0, 0, -29.0])