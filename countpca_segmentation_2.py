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
        self.slice_count = 10
        self.origin_points = 6
        self.most_combination_points = 25
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.filter_rate = 0.05
        self.ori_adarate = 0.5
        self.step_adarate = 1.5
        self.max_adarate = 1.1
        self.pointsame_threshold = 0.5
        self.split_by_count = True

def surf_segmentation(points, config, paint_when_end = False):
    global ELAPSE_SEG
    config.slice_count = min(int(len(points) / config.origin_points), config.slice_count)
    assert len(points) / config.slice_count >= config.origin_points
    adasurconfig = AdaSurfConfig({'origin_points': config.origin_points
        , 'most_combination_points': config.most_combination_points
        , 'same_threshold': config.same_threshold
        , 'filter_rate': config.filter_rate
        , 'ori_adarate': config.ori_adarate
        , 'step_adarate': config.step_adarate
        , 'max_adarate': config.max_adarate
        , 'pointsame_threshold': config.pointsame_threshold
        })
    surfs = []
    slice_fig = []
    npoints = point_normalize(points)
    starttime = time.clock()

    pca_md = mlab.PCA(np.copy(npoints))
    projection0 = pca_md.Y[:, 0]

    if config.split_by_count:
        step_count = len(projection0) / config.slice_count
        pointsets = [np.array([]).reshape(0,3)] * config.slice_count
        sorted_projection0_index = np.argsort(projection0)
        current_slot_count, ptsetid = 0, 0
        for index in sorted_projection0_index:
            pointsets[ptsetid] = np.vstack((pointsets[ptsetid], npoints[index]))
            current_slot_count += 1
            if current_slot_count > step_count:
                current_slot_count = 0
                ptsetid += 1
    else:
        projection0min, projection0max = np.min(projection0), np.max(projection0)
        step_len = (projection0max - projection0min) / config.slice_count
        pointsets = [np.array([]).reshape(0,3)] * config.slice_count
        for i in xrange(len(projection0)):
            if projection0[i] == projection0min:
                ptsetid = config.slice_count - 1
            else:
                ptsetid = int((projection0[i] - projection0min) / step_len)
            pointsets[ptsetid] = np.vstack((pointsets[ptsetid], npoints[index]))

    partial_surfs, fail = [], np.array([]).reshape(0,3)

    for (ptset, ptsetindex) in zip(pointsets, range(len(pointsets))):
        print "--------------------------------------"
        print "before segment", ptsetindex, '/', len(pointsets)
        allptfortest = np.vstack((ptset, np.array(fail)))
        print "len of surf is: ", len(partial_surfs), ", len of points is: ", len(allptfortest)
        if allptfortest != None and len(allptfortest) > 0 :
            partial_surfs, _, fail, extradata = identifysurf(allptfortest, adasurconfig, donorm = False, surfs = partial_surfs, title = str(ptsetindex), paint_when_end = paint_when_end)
            if paint_when_end:
                slice_fig.append(extradata[0])
        if fail == None:
            print "after segment", ptsetindex, "len of surf", len(partial_surfs), "fail is None", fail
        else:
            print "after segment", ptsetindex, "len of surf", len(partial_surfs), "len of fail", len(fail)
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

    return surfs, npoints, (slice_fig, )

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')
    config = SurfSegConfig()
    print 'config', config.__dict__
    import time
    starttime = time.clock()
    surfs, npoints, extradata = surf_segmentation(c, config, paint_when_end = False)

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
    print '----------BELOW ARE POINTS----------'
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[2]
    paint_surfs(surfs, npoints, 'all')
    print extradata
    for slice_fig in extradata[0]:
        slice_fig.show()


