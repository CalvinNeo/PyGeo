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
from new_adasurf import AdaSurfConfig, adasurf, paint_surfs, identifysurf, point_normalize, Surface


ELAPSE_SEG = 0
class NewConfig:
    def __init__(self):
        self.slice_count = 4
        self.origin_points = 5
        self.most_combination_points = 35
        self.same_threshold = 0.05 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.pointsame_threshold = 1.0
        self.filter_rate = 0.08
        self.filter_count = 80
        self.ori_adarate = 2.0
        self.step_adarate = 1.0
        self.max_adarate = 2.0
        self.split_by_count = True
        self.weak_abort = 70

def paint_points(points, show = True, title = '', xlim = None, ylim = None, zlim = None):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    if xlim == None:
        xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
    if ylim == None:
        ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
    if zlim == None:
        zlim = (np.min(points[:, 2]), np.max(points[:, 2]))
    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]
    ax.scatter(x1, y1, z1, c='r')

    ax.set_zlim(zlim[0], zlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    pl.title(title)
    if show:
        pl.show()
    return fig

def newsln(points, config, paint_when_end = False):
    global ELAPSE_SEG
    assert len(points) / config.slice_count >= config.origin_points
    adasurconfig = AdaSurfConfig({'origin_points': config.origin_points
        , 'most_combination_points': config.most_combination_points
        , 'same_threshold': config.same_threshold
        , 'filter_rate': config.filter_rate
        , 'ori_adarate': config.ori_adarate
        , 'step_adarate': config.step_adarate
        , 'max_adarate': config.max_adarate
        , 'pointsame_threshold': config.pointsame_threshold
        , 'filter_count' : config.filter_count
        , 'weak_abort' : config.weak_abort
        })
    surfs = []
    slice_fig = []
    npoints = point_normalize(points)
    starttime = time.clock()
    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))


    partial_surfs, allptfortest = [], npoints.copy()
    # for (ptset, ptsetindex) in zip(pointsets, range(len(pointsets))):
    for iter_times in xrange(5):
        print "--------------------------------------"
        print "before iter", iter_times
        print "len of surf is: ", len(partial_surfs), ", len of points is: ", len(allptfortest)
        if allptfortest != None and len(allptfortest) > 0 :
            partial_surfs, _, fail, extradata = identifysurf(allptfortest, adasurconfig, donorm = False, surfs = partial_surfs, title = str(iter_times)
                , paint_when_end = paint_when_end, current_direction = None)
            if paint_when_end:
                slice_fig.append(extradata[0])
            allptfortest = fail
        for x in partial_surfs:
            x.printf()

    surfs.extend(partial_surfs)

    return surfs, npoints, (slice_fig, )

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')
    config = NewConfig()
    print 'config', config.__dict__
    import time
    starttime = time.clock()
    surfs, npoints, extradata = newsln(c, config, paint_when_end = True)

    print "----------BELOW ARE SURFACES---------- count:", len(surfs)
    print 'TOTAL: ', time.clock() - starttime
    print 'ELAPSE_SEG: ', ELAPSE_SEG
    ALL_POINT = 0
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s.args # surface args
        print s.residuals # MSE
        print len(s.points)
        ALL_POINT += len(s.points)
        # print s[2] # npoints
        print '**************************************'
    print 'ALL_POINT: ', ALL_POINT
    print '----------BELOW ARE POINTS----------'
    # for s,i in zip(surfs, range(len(surfs))):
    #     print "SURFACE ", i
    #     print s.points
    paint_surfs(surfs, npoints, 'all')
    # print extradata
    # for slice_fig in extradata[0]:
    #     slice_fig.show()


