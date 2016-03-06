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
        self.slice_count = 4
        self.origin_points = 5
        self.most_combination_points = 35
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.filter_rate = 0.01
        self.ori_adarate = 1.0

def surf_segmentation(points, config):
    global ELAPSE_SEG
    config.slice_count = min(int(len(points) / config.origin_points), config.slice_count)
    assert len(points) / config.slice_count >= config.origin_points
    surfs = []
    npoints = point_normalize(points)

    pca_md = mlab.PCA(np.copy(npoints))
    projection0 = pca_md.Y[:, 0]

    step_count = len(projection0) / config.slice_count
    pointsets = [np.array([]).reshape(0,3)] * config.slice_count
    starttime = time.clock()

    # # projection0_index = np.hstack((projection0, np.arange(len(projection0))))
    # sorted_projection0_index = np.argsort(projection0)
    # # sorted_projection0 = projection0[sorted_projection0_index]
    # current_slot_count, ptsetid = 0, 0
    # # for (index, value) in zip(sorted_projection0_index, sorted_projection0):
    # for index in sorted_projection0_index:
    #     pointsets[ptsetid] = np.vstack((pointsets[ptsetid], npoints[index]))
    #     current_slot_count += 1
    #     if current_slot_count > step_count:
    #         current_slot_count = 0
    #         ptsetid += 1

    sorted_projection0_index = np.argsort(npoints[:, 0])
    current_slot_count, ptsetid = 0, 0
    for index in sorted_projection0_index:
        pointsets[ptsetid] = np.vstack((pointsets[ptsetid], npoints[index]))
        current_slot_count += 1
        if current_slot_count > step_count:
            current_slot_count = 0
            ptsetid += 1

    partial_surfs = []
    for ptset in pointsets:
        print "before segment", len(partial_surfs), len(ptset)
        if len(ptset) > 0:
            partial_surfs, _ = identifysurf(np.copy(ptset), AdaSurfConfig(
                {'origin_points': config.origin_points, 'most_combination_points': config.most_combination_points, 'same_threshold': config.same_threshold, 'filter_rate': config.filter_rate, 'ori_adarate': config.ori_adarate
                }), donorm = False, surfs = partial_surfs)
        print "after segment", len(partial_surfs)
    surfs.extend(partial_surfs)

    # fig = pl.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(npoints[:, 0], npoints[:, 1], npoints[:, 2], c='r')
    # x = np.linspace(0, pca_md.Wt[0, 0] * 100, 300)
    # y = np.linspace(0, pca_md.Wt[0, 1] * 100, 300)
    # z = np.linspace(0, pca_md.Wt[0, 2] * 100, 300)
    # ax.plot(x, y, z, c='k')
    # x = np.linspace(0, pca_md.Wt[1, 0] * 100, 300)
    # y = np.linspace(0, pca_md.Wt[1, 1] * 100, 300)
    # z = np.linspace(0, pca_md.Wt[1, 2] * 100, 300)
    # ax.plot(x, y, z, c='g')
    # pl.show()

    return surfs, npoints

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints = surf_segmentation(c, SurfSegConfig())
    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))
    print "----------BELOW ARE SURFACES---------- count:", len(surfs)
    print 'TOTAL: ', time.clock() - starttime
    print 'ELAPSE_SEG: ', ELAPSE_SEG
    ALL_POINT = 0
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[0] # surface args
        print s[1] # MSE
        ALL_POINT += len(s[2])
        print len(s[2])
        # print s[2] # npoints
        print '**************************************'
    print 'ALL_POINT: ', ALL_POINT

    paint_surfs(surfs, npoints, xlim, ylim, zlim)

    # c = point_normalize(c)
    # fig = pl.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x1 = c[:, 0]
    # y1 = c[:, 1]
    # z1 = c[:, 2]
    # # tan_color = np.ones((len(x1), len(y1))) * np.arctan2(len(surfs)) # c='crkgmycrkgmycrkgmycrkgmy'[surf_id]
    # ax.scatter(x1, y1, z1, c='c', marker='o')
    # pl.show()
