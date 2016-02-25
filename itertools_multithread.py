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
import threading
import time

from scipy.optimize import leastsq

if __name__ == '__main__':
    cb = combinations([1,2,3,4,5], 3)
    def loop():
        for x in cb:
            print x
            time.sleep(1)

    t1 = threading.Thread(target=loop, name='LoopThread1')
    t2 = threading.Thread(target=loop, name='LoopThread2')

    t1.start()
    t2.start()
    t1.join()
    t2.join()
