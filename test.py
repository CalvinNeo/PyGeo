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

if __name__ == '__main__':
    # arr = np.arange(9).reshape((3, 3))
    # print arr
    # np.random.shuffle(arr[:])
    # print arr

    arr = np.arange(9)
    print arr[0:255]