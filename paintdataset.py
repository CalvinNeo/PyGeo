#coding:utf8

import numpy as np, scipy
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import cm
from matplotlib import mlab
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from itertools import *
import collections
from multiprocessing import Pool
import random
from scipy.optimize import leastsq
from adasurf import AdaSurfConfig, adasurf, paint_surfs, identifysurf, point_normalize


if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')
    c = point_normalize(c)
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = c[:, 0]
    y1 = c[:, 1]
    z1 = c[:, 2]
    # tan_color = np.ones((len(x1), len(y1))) * np.arctan2(len(surfs)) # c='crkgmycrkgmycrkgmycrkgmy'[surf_id]
    ax.scatter(x1, y1, z1, c='c', marker='o')
    pl.show()