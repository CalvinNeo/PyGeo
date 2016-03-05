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

class SurfSegConfig:
    def __init__(self):
        self.slice_count = 5


def surf_segmentation(points, config):
    npoints = point_normalize(points)
    surfs = []
    # cov = np.cov(npoints)
    pca_md = mlab.PCA(npoints)
    projection0 = pca_md.Y[:, 0]
    slice_step = projection0[-1, 0] - projection0[0, 0]
    print np.std(pca_md.Y[:, 0]),np.std(pca_md.Y[:, 1]),np.std(pca_md.Y[:, 2])
    # print projection0.shape, npoints.shape
    # print np.linalg.norm(pca_md.Wt[0])
    # fig = pl.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(npoints[:, 0], npoints[:, 1], npoints[:, 2], c='r')
    # x = np.linspace(0, pca_md.Wt[0, 0], 300)
    # y = np.linspace(0, pca_md.Wt[0, 1], 300)
    # z = np.linspace(0, pca_md.Wt[0, 2], 300)
    # ax.plot(x, y, z, c='k')
    # x = np.linspace(0, pca_md.Wt[1, 0], 300)
    # y = np.linspace(0, pca_md.Wt[1, 1], 300)
    # z = np.linspace(0, pca_md.Wt[1, 2], 300)
    # ax.plot(x, y, z, c='g')
    # pl.show()
    return surfs, npoints

if __name__ == '__main__':
    c = np.loadtxt('4.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints = surf_segmentation(c, SurfSegConfig())
    # print "----------BELOW ARE SURFACES----------"
    # for s,i in zip(surfs, range(len(surfs))):
    #     print "SURFACE ", i
    #     print s[0]
    #     print s[1]
    #     print s[2]
    #     print '**************************************'

    # print len(surfs)


    # xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    # ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    # zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))

    # paint_surfs(surfs, npoints, xlim, ylim, zlim)
