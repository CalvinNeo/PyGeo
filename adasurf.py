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

class AdaSurfConfig:
    def __init__(self):
        self.origin_points = 5
        self.most_combination_points = 35
        self.same_threshold = 0.5

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

def adasurf(points, config):
    global ELAPSE_LSQ
    def residuals(params, x, y, z, regularization = 0.0):
        rt = z - config.surf_fun(x, y, params)
        rt = np.append(rt, np.sqrt(regularization)*params)
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

    return r[0], MSE(r[0], points), points

def paint_surfs(surfs, points, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.1, 1.1), show = True):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ans, surf_id in zip(surfs, range(len(surfs))):
        a, b, c = ans[0][0], ans[0][1], ans[0][2]
        X = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.0)
        Y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.0)
        X, Y = np.meshgrid(X, Y)
        Z = -(X*a + Y*b + c)
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # fig.colorbar(s, shrink=0.5, aspect=5)
        s = ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15)
        x1 = ans[2][:, 0]
        y1 = ans[2][:, 1]
        z1 = ans[2][:, 2]
        # tan_color = np.ones((len(x1), len(y1))) * np.arctan2(len(surfs)) # c='crkgmycrkgmycrkgmycrkgmy'[surf_id]
        ax.scatter(x1, y1, z1, c='crgkmycrgkmycrgkmy'[surf_id])

    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if show:
        pl.show()

def Pipecycle(iterable, predicate, roundclearup = None):
    prev = None
    while len(iterable) > 0:
        fail = []
        for x in iterable:
            val = predicate(x)
            if not val:
                fail.append(x)
        iterable = np.array(fail)
        if roundclearup(iterable):
            return
        print 'points remaining', len(iterable)
        if prev == len(iterable):
            return
        else:
            prev = len(iterable)

def identifysurf(points, config, donorm = True):
    def same_surf(surf, point):
        # print abs(point[2]-config.surf_fun(point[0], point[1], surf[0])) , surf[1] * 100
        e = abs(point[2]-config.surf_fun(point[0], point[1], surf[0]))
        return e <= config.same_threshold * nstd, e

    def new_surf(partial_points):
        '''
            return True: all points are fitted, Pipecycle quit loop
            return False: Pipecycle should loop again and fit points
        '''
        global ELAPSE_STD
        all_surf = []
        starttime = time.clock()
        adaptive_rate = 1.0
        
        np.random.shuffle(partial_points[:])
        for group_id in xrange(int(math.ceil(len(partial_points)*1.0/config.most_combination_points))):
            while len(all_surf) == 0: # 如果始终不能生成新的面
                # choices = random.sample(partial_points, min(config.most_combination_points, len(partial_points)))
                choices = partial_points[group_id*config.most_combination_points:(group_id+1)*config.most_combination_points, :]
                for circum in combinations(choices, config.origin_points):
                    # 当取得的点的标准差小于总体的标准差才进行最小二乘拟合
                    starttime_circum = time.clock()
                    std_circum = np.std(np.array(circum))
                    ELAPSE_STD += time.clock() - starttime_circum
                    if std_circum < config.same_threshold * nstd * adaptive_rate:
                        generated_surf = adasurf(np.array(circum), config)
                        if generated_surf[1] < config.same_threshold * nstd:
                            all_surf.append(generated_surf)
                print 'new_surface: elapse', time.clock() - starttime, 'surface_count', len(all_surf), 'adaptive_rate', adaptive_rate
                if len(all_surf) > 0: # 如果生成了若干新面
                    surfs.append(min(all_surf, key=lambda x:x[1]))
                    return False
                else:
                    if len(partial_points) <= config.origin_points: # 如果剩余的点数小于生成平面的基点数
                        return True
                    else: # 如果剩余的点数大于生成平面的基点数，说明是在标准差阶段卡住了，适当地提高标准差的限制，继续跑
                        adaptive_rate *= 2

    def judge_point(point):
        suitable_surfs = []
        for surf, surf_id in zip(surfs, range(len(surfs))):
            pre, e = same_surf(surf, point)
            if pre:
                suitable_surfs.append((surf, e, surf_id))
        if len(suitable_surfs) > 0:
            surf, _, surf_id = min(suitable_surfs, key=lambda x:x[1])
            surfs[surf_id] = adasurf(np.vstack((surf[2],point)), config)
            return True
        else:
            return False

    surfs = []

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points
    nstd = np.std(npoints)
    print 'nstd', nstd
    Pipecycle(npoints, judge_point, new_surf)

    return surfs, npoints

if __name__ == '__main__':
    c = np.loadtxt('4.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints = identifysurf(c, AdaSurfConfig())
    print 'TOTAL: ', time.clock() - starttime
    print "ELAPSE_LSQ: ", ELAPSE_LSQ
    print "ELAPSE_STD: ", ELAPSE_STD
    print "----------BELOW ARE SURFACES----------"
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[0]
        print s[1]
        print s[2]
        print '**************************************'

    print len(surfs)
    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))

    paint_surfs(surfs, npoints, xlim, ylim, zlim)
