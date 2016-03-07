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
        self.ori_adarate = 0.5
        self.step_adarate = 1.5
        self.max_adarate = 1.5
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

def Pipecycle(iterable, predicate, roundclearup = None, clearing = None):
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
            fail = clearing(fail)
            return fail
        print 'after adding a new surface, unfitted points remaining', len(iterable)
        assert prev != len(iterable)
        if prev == len(iterable):
            # assert before return
            return
        else:
            prev = len(iterable)

def identifysurf(points, config, donorm = True, surfs = []):
    def same_surf(surf, point):
        e = abs(point[2]-config.surf_fun(point[0], point[1], surf[0]))
        return e <= config.same_threshold * nstd, e

    def new_surf(partial_points):
        '''
            return True: all points are fitted, Pipecycle quit loop; or there's no fitting surface
            return False: Pipecycle should loop again and fit points
            dependencies: points
        '''
        global ELAPSE_STD
        all_surf = []
        import time
        starttime = time.clock()
        adaptive_rate = config.ori_adarate
        
        np.random.shuffle(partial_points[:])
        len_group = int(math.ceil(len(partial_points)*1.0/config.most_combination_points))
        for group_id in xrange(len_group):
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
                print 'try_new_surface: elapse', time.clock() - starttime,'group_id', group_id, '/', len_group, 'surface_count', len(all_surf), 'adaptive_rate', adaptive_rate, 'npartial_points', len(partial_points)
                # print len(sorted(all_surf, reverse = True, cmp = lambda x,y: len(x[2]) > len(y[2]))[-1][2]), config.filter_rate * len(points)
                # all_surf = filter(lambda x: len(x) > config.filter_rate * len(points), all_surf)
                if len(all_surf) > 0: # 如果生成了若干新面
                    surfs.append(min(all_surf, key=lambda x:x[1]))
                    return False
                else:
                    if len(partial_points) <= config.origin_points: # 如果剩余的点数小于生成平面的基点数
                        return True
                    else: # 如果剩余的点数大于生成平面的基点数，说明是在标准差阶段卡住了，适当地提高标准差的限制，继续跑
                        if adaptive_rate < config.max_adarate:
                            if adaptive_rate * config.step_adarate < config.max_adarate:
                                adaptive_rate *= config.step_adarate
                            else:
                                adaptive_rate = config.max_adarate * 1.01
                        else: # adarate不能过大，否则就不精确了
                            return True


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

    def remove_poor_support_surface(fail):
        newfail = np.array([]).reshape(0, 3)
        index_to_remove = []
        for (surf, index) in zip(surfs, range(len(surfs))):
            supporting = len(surf[2])
            if supporting < 40: # config.filter_rate * len(points): # if this surface is poor supported
                # remove the surf and add its supporting points back to fail
                print "***drop one"
                newfail = np.vstack((newfail, surf[2]))
                index_to_remove.append(index)
        for index in sorted(index_to_remove, reverse=True):
            del surfs[index]
        return np.vstack((fail, newfail))

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points

    nstd = np.std(npoints)
    print 'nstd of all points in this segment', nstd
    fail = Pipecycle(npoints, judge_point, new_surf, remove_poor_support_surface)

    return surfs, npoints, fail

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints, _ = identifysurf(c, AdaSurfConfig())
    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))
    print 'TOTAL: ', time.clock() - starttime
    print "ELAPSE_LSQ: ", ELAPSE_LSQ
    print "ELAPSE_STD: ", ELAPSE_STD
    '''
        考虑到生成面的时候点并没有被移除，所以点可能变多
    '''
    print 'TOTAL_POINT: ', len(npoints)
    print "----------BELOW ARE SURFACES----------"
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print s[0] # surface args
        print s[1] # MSE
        print len(s[2])
        # print s[2] # npoints
        print '**************************************'

    print len(surfs)


    paint_surfs(surfs, npoints, xlim, ylim, zlim)
