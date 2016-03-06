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

def paint_surfs(surfs, points, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.1, 1.1), show = True):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ans, surf_id in zip(surfs, range(len(surfs))):
        a, b, c = ans[0][0], ans[0][1], ans[0][2]
        X = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.0)
        Y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.0)
        X, Y = np.meshgrid(X, Y)
        Z = -(X*a + Y*b + c)
        s = ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15)
        x1 = ans[2][:, 0]
        y1 = ans[2][:, 1]
        z1 = ans[2][:, 2]
        # tan_color = np.ones((len(x1), len(y1))) * np.arctan2(len(surfs)) # c='crkgmycrkgmycrkgmycrkgmy'[surf_id]
        # ax.scatter(x1, y1, z1, c='rcykgm'[surf_id % 6], marker='o^sd*+xp'[int(surf_id/6)])

    ax.set_zlim(zlim[0], zlim[1])
    # ax.set_ylim(ylim[0], ylim[1])
    # ax.set_xlim(xlim[0], xlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    if show:
        pl.show()

class AdaSurfConfig:
    def __init__(self, *initial_data, **kwargs):
        self.origin_points = 5
        self.most_combination_points = 20
        self.same_threshold = 0.5
        self.filter_rate = 0.01
        self.ori_adarate = 1.0
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

def Pipecycle(iterable, predicate, roundclearup = None):
    '''
        In this case:
            predicate -- judge_point
    '''
    prev = None
    while len(iterable) > 0:
        fail = []
        for x in iterable:
            val = predicate(x)
            if not val:
                fail.append(x)
        iterable = np.array(fail) # renew iterable
        if roundclearup(iterable):
            return
        print 'points remaining', len(iterable)
        assert prev != len(iterable)
        if prev == len(iterable):
            return
        else:
            prev = len(iterable)

def identifysurf(points, config, donorm = True, surfs = []):
    def same_surf(surf, point):
        # print abs(point[2]-config.surf_fun(point[0], point[1], surf[0])) , surf[1] * 100
        e = abs(point[2]-config.surf_fun(point[0], point[1], surf[0]))
        return e <= config.same_threshold * nstd, e

    def new_surf(partial_points):
        '''
            return True: all points are fitted, Pipecycle quit loop
            return False: Pipecycle should loop again and fit points
            dependencies: points
        '''
        global ELAPSE_STD
        all_surf = []
        import time
        starttime = time.clock()
        adaptive_rate = config.ori_adarate
        
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
                            # 这里generated_surf里面已经包含了生成的点，但是这些点还没有从npoints中被移除，所以结果里面点会变多
                            all_surf.append(generated_surf)
                print 'new_surface: elapse', time.clock() - starttime, 'surface_count', len(all_surf), 'adaptive_rate', adaptive_rate, 'npartial_points', len(partial_points)
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
            # renew surf
            surfs[surf_id] = adasurf(np.vstack((surf[2],point)), config)
            return True
        else:
            return False

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points
    nstd = np.std(npoints)
    print 'nstd', nstd
    Pipecycle(npoints, judge_point, new_surf)

    return surfs, npoints

ELAPSE_SEG = 0
class SurfSegConfig:
    def __init__(self):
        self.slice_count = 10
        self.origin_points = 6
        self.most_combination_points = 25
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.filter_rate = 0.01
        self.ori_adarate = 1.0

def surf_segmentation(points, config):
    global ELAPSE_SEG
    config.slice_count = min(int(len(points) / config.origin_points), config.slice_count)
    assert len(points) / config.slice_count >= config.origin_points
    surfs = []
    npoints = point_normalize(points)
    # cov = np.cov(npoints)
    pca_md = mlab.PCA(np.copy(npoints))
    projection0 = pca_md.Y[:, 0]
    step_count = len(projection0) / config.slice_count
    pointsets = [np.array([]).reshape(0,3)] * config.slice_count
    starttime = time.clock()

    # projection0_index = np.hstack((projection0, np.arange(len(projection0))))
    sorted_projection0_index = np.argsort(projection0)
    # sorted_projection0 = projection0[sorted_projection0_index]
    current_slot_count, ptsetid = 0, 0
    # for (index, value) in zip(sorted_projection0_index, sorted_projection0):
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
