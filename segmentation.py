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
from adasurf import AdaSurfConfig, adasurf, paint_surfs, identifysurf, point_normalize

class SurfSegConfig:
    def __init__(self):
        pass


def surf_segmentation(points, config):
    npoints = point_normalize(points)
    

if __name__ == '__main__':
    c = np.loadtxt('4.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints = surf_segmentation(c, SurfSegConfig())
    print "----------BELOW ARE SURFACES----------"
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[0]
        print s[1]
        print s[2]
        print '**************************************'

    print len(surfs)
    # xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    # ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    # zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))

    # paint_surfs(surfs, npoints, xlim, ylim, zlim)
