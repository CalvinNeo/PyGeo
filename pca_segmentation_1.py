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


ELAPSE_SEG = 0
class SurfSegConfig:
    def __init__(self):
        self.slice_count = 2
        self.origin_points = 5
        self.most_combination_points = 30
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.filter_rate = 0.01

def surf_segmentation(points, config):
    global ELAPSE_SEG
    assert len(points) / config.slice_count >= config.origin_points
    npoints = point_normalize(points)
    # cov = np.cov(npoints)
    pca_md = mlab.PCA(npoints)
    projection0 = pca_md.Y[:, 0]
    projection0min, projection0max = np.min(projection0), np.max(projection0)
    slice_step = (projection0max - projection0min) / config.slice_count
    pointsets = [np.array([]).reshape(0,3)] * config.slice_count
    surfs = []
    starttime = time.clock()
    for row_id in xrange(len(projection0)): 
        if projection0[row_id] == projection0max:
            ptsetid = config.slice_count - 1
        else:
            ptsetid = int((projection0[row_id]-projection0min) / slice_step)
        pointsets[ptsetid] = np.vstack((pointsets[ptsetid], npoints[row_id]))
    ELAPSE_SEG += time.clock() - starttime
    partial_surfs = []
    for ptset in pointsets:
        print "before segment", len(partial_surfs)
        if len(ptset) > 0:
            partial_surfs, _ = identifysurf(ptset, AdaSurfConfig(
                {'origin_points': config.origin_points, 'most_combination_points': config.most_combination_points, 'same_threshold': config.same_threshold}), donorm = False, surfs = partial_surfs)
            # # 注意这里不能简单地extend，应当将surfs和partial_surfs去重
            # if len(partial_surfs) > 0:
            #     surfs.extend(partial_surfs)
        print "after segment", len(partial_surfs)
    surfs.extend(partial_surfs)
    # print np.std(pca_md.Y[:, 0]),np.std(pca_md.Y[:, 1]),np.std(pca_md.Y[:, 2])
    # print pca_md.Y[:, 0]
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
    c = np.loadtxt('5.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints = surf_segmentation(c, SurfSegConfig())
    print "----------BELOW ARE SURFACES---------- count:", len(surfs)
    print 'TOTAL: ', time.clock() - starttime
    print 'ELAPSE_SEG: ', ELAPSE_SEG

    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[0] # surface args
        print s[1] # MSE
        print len(s[2])
        # print s[2] # npoints
        print '**************************************'

    # print len(surfs)

    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))

    paint_surfs(surfs, npoints, xlim, ylim, zlim)
