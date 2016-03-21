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
import time

class Surface:
    def __init__(self, *initial_data, **kwargs):
        self.residuals = 0
        self.points = np.array([]).reshape(0, 3)
        self.args = np.array([1.0,1.0,1.0])
        self.initial_points = np.array([]).reshape(0, 3)

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


    def printf(self):
        print str((self.args, self.residuals, self.initial_points))

    def addpoint(self, p):
        self.points = np.vstack((self.points, p))

    def normalizer(self):
        return math.sqrt(self.args[0]**2 + self.args[1]**2 + 1)

    def direct(self):
        return np.array([self.args[0], self.args[1], 1])

    def normvec(self):
        return self.direct() / self.normalizer()

    def __str__(self):
        return str(tuple(self.args, self.residuals, self.points, self.initial_points))

def paint_surfs(surfs, points, show = True, title = ''):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
    ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
    zlim = (np.min(points[:, 2]), np.max(points[:, 2]))
    for ans, surf_id in zip(surfs, range(len(surfs))):
        a, b, c = ans.args[0], ans.args[1], ans.args[2]
        X = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.0)
        Y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.0)
        X, Y = np.meshgrid(X, Y)
        Z = -(X*a + Y*b + c)
        s = ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15)
        x1 = ans.points[:, 0]
        y1 = ans.points[:, 1]
        z1 = ans.points[:, 2]
        ax.scatter(x1, y1, z1, c='rcykgm'[surf_id % 6], marker='o^sd*+xp'[int(surf_id/6)])

    ax.set_zlim(zlim[0], zlim[1])
    # ax.set_ylim(ylim[0], ylim[1])
    # ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    pl.title(title)
    if show:
        pl.show()
    return fig

class AdaSurfConfig:
    def __init__(self, *initial_data, **kwargs):
        self.origin_points = 7
        self.most_combination_points = 20
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.pointsame_threshold = 0.1
        self.filter_rate = 0.04
        self.filter_count = 10
        self.ori_adarate = 1.0
        self.step_adarate = 1.0
        self.max_adarate = 1.0
        self.weak_abort = 20

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    # 待拟合面的函数，x是变量，p是参数
    def surf_fun(self, x, y, params):
        a, b, c = params
        return -(a*x + b*y + c)

ELAPSE_LSQ = 0
ELAPSE_STD = 0

def point_normalize(points):
    points = np.array(points)
    points[:, 0] = points[:, 0] - np.mean(points[:, 0])
    points[:, 1] = points[:, 1] - np.mean(points[:, 1])
    points[:, 2] = points[:, 2] - np.mean(points[:, 2])
    return points

def adasurf(points, config, initial_points = None):
    global ELAPSE_LSQ
    def residuals(params, x, y, z, regularization = 0.0):
        rt = z - config.surf_fun(x, y, params)
        # rt = np.append(rt, np.sqrt(regularization)*params)
        return rt

    def MSE(params, points):
        e = (points[:,2] - config.surf_fun(points[:,0], points[:,1], params))
        return np.sqrt(np.dot(e.T, e)/len(e))

    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]

    starttime = time.clock()
    r = leastsq(residuals, [1, 0.5, 1], args=(x1, y1, z1))
    ELAPSE_LSQ += time.clock() - starttime
    
    return Surface(args = r[0] , residuals = MSE(r[0], points) , points = points, initial_points = initial_points)


PRINT_COUNT = 0
def identifysurf(points, config, donorm = True, surfs = [], paint_when_end = False, title = '', current_direction = None):
            
    def same_surf(surf1, surf2):
        v1, v2 = surf1.normvec(), surf2.normvec()
        return np.dot(v1, v2) / (v1.norm() * v2.norm()) < 0.1

    def combine_surf():
        # combine similar surf
        pass

    def belong_point(unfixed_point):
        # if possible, add one point to one of the surfaces
        pass

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points

    print 'len(surf)', len(surfs)

    all_surf = []
    partial_points = npoints.copy()
    np.random.shuffle(partial_points[:])
    len_group = int(math.ceil(len(partial_points)*1.0/config.most_combination_points))
    for group_id in xrange(len_group):
        choices = partial_points[group_id*config.most_combination_points:(group_id+1)*config.most_combination_points, :]
        generated_surf = adasurf(np.array(circum), config)
        if generated_surf.residuals < config.same_threshold:
            all_surf.append(generated_surf)


    fig = None
    if paint_when_end:
        fig = paint_surfs(surfs, npoints, title = title, show = False)

    return surfs, npoints, (fig, )

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')

    import time
    surfs, npoints, _ = identifysurf(c, AdaSurfConfig())
    print 'TOTAL-TIME: ', time.clock() - starttime
    print 'TOTAL_POINT: ', len(npoints)
    print "----------BELOW ARE SURFACES----------"
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s.args # surface args
        print s.residuals # MSE
        print len(s.points)
        # print s[2] # npoints
        print '**************************************'

    print len(surfs)


    paint_surfs(surfs, npoints)
