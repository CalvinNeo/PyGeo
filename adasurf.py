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

        def __str__(self):
            return str(tuple(self.args, self.residuals, self.points, self.initial_points))

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

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
        # ax.scatter(x1, y1, z1, c='rcykgm'[surf_id % 6], marker='o^sd*+xp'[int(surf_id/6)])

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
        self.slice_count = 1
        self.origin_points = 7
        self.most_combination_points = 20
        self.same_threshold = 0.5 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.pointsame_threshold = 0.1
        self.filter_rate = 0.04
        self.ori_adarate = 1.0
        self.step_adarate = 1.0
        self.max_adarate = 1.0

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

    return Surface(args = r[0], residuals = MSE(r[0], points), points = points, initial_points = initial_points)

def Pipecycle(iterable, predicate, roundclearup = None, clearing = None):
    '''
        In this case:
            predicate -- judge_point
            rountclearup -- new_surf
    '''
    prev = None
    while len(iterable) > 0:
        fail = []
        for x in iterable:
            val = predicate(x)
            if not val:
                fail.append(x)
        iterable = np.array(fail) # renew iterable
        print 'before adding a new surface, unfitted points remaining', len(iterable)
        if roundclearup(iterable):
            fail = clearing(fail)
            return fail
        # assert prev != len(iterable)
        if prev == len(iterable):
            # assert before return
            return
        else:
            prev = len(iterable)

def identifysurf(points, config, donorm = True, surfs = [], paint_when_end = False, title = ''):
    def same_surf(surf, point):
        # e = abs(point[2]-config.surf_fun(point[0], point[1], surf.args))
        A = np.array([surf.args[0], surf.args[1], -1, surf.args[2]]).reshape(1, 4)
        X = np.array([point[0], point[1], point[2], 1]).reshape(4, 1)
        upper = np.dot(A, X)[0,0]
        lower = math.sqrt(np.dot(A[0:3], A[0:3].reshape(4,1)))
        e = abs(upper / lower)
        # print e.shape, upper.shape, nstd, e[0][0].shape, e[0][0] <= config.pointsame_threshold * nstd, config.pointsame_threshold
        # return e <= config.pointsame_threshold * nstd, e
        return e <= config.pointsame_threshold, e


    def new_surf(partial_points):
        '''
            return True: all points are fitted, Pipecycle quit loop; or there's no fitting surface
            return False: Pipecycle should loop again and fit points
            dependencies: points
        '''
        # renew surfs
        # for surf_id in xrange(len(surfs)):
        #     surfs[surf_id] = adasurf(surfs[surf_id].points, config)

        global ELAPSE_STD
        TOP_MIN_STD = 99999
        all_surf = []
        import time
        starttime = time.clock()
        adaptive_rate = config.ori_adarate
        
        np.random.shuffle(partial_points[:])
        len_group = int(math.ceil(len(partial_points)*1.0/config.most_combination_points))
        while len(all_surf) == 0: # 如果始终不能生成新的面
            for group_id in xrange(len_group):
                choices = partial_points[group_id*config.most_combination_points:(group_id+1)*config.most_combination_points, :]
                for circum in combinations(choices, config.origin_points):
                    # 当取得的点的标准差小于总体的标准差才进行最小二乘拟合
                    starttime_circum = time.clock()
                    std_circum = np.std(np.array(circum)[:, 1:-1])
                    TOP_MIN_STD = min(TOP_MIN_STD, std_circum)
                    ELAPSE_STD += time.clock() - starttime_circum
                    if std_circum < config.same_threshold * nstd * adaptive_rate: # 如果方差满足要求
                        generated_surf = adasurf(np.array(circum), config)
                        if generated_surf.residuals < config.same_threshold * nstd:
                            # 这里generated_surf里面已经包含了生成的点，但是这些点还没有从npoints中被移除，所以结果里面点会变多
                            all_surf.append(generated_surf)
                        else:
                            pass
                            # print "ada gameover"
                print 'try_new_surface: elapse', time.clock() - starttime,'group_id', group_id, '/', len_group, 'surface_count', len(all_surf), 'adaptive_rate', adaptive_rate, 'npartial_points', len(partial_points)
            
            if len(all_surf) > 0: # 如果生成了若干新面
                surfs.append(min(all_surf, key=lambda x:x.residuals))
                return False
            else:
                if len(partial_points) <= config.origin_points: # 如果剩余的点数小于生成平面的基点数，这应该可以在之前判定的
                    print 'less then happen'
                    return True
                else: # 如果剩余的点数大于生成平面的基点数，说明是在标准差阶段卡住了，适当地提高标准差的限制，继续跑
                    if adaptive_rate < config.max_adarate:
                        if adaptive_rate * config.step_adarate < config.max_adarate:
                            adaptive_rate *= config.step_adarate
                        else:
                            adaptive_rate = config.max_adarate * 1.01
                    else: # adarate不能过大，否则就不精确了
                        print 'TOP_MIN_STD', TOP_MIN_STD, nstd
                        return True


    def judge_point(point):
        suitable_surfs = []
        for surf, surf_id in zip(surfs, range(len(surfs))):
            pre, e = same_surf(surf, point)
            if pre:
                suitable_surfs.append((surf, e, surf_id))
        if len(suitable_surfs) > 0:
            surf, _, surf_id = min(suitable_surfs, key=lambda x:x[1])
            # NO renew surf
            surfs[surf_id] = Surface(args = surf.args, residuals = surf.residuals, points = np.vstack((surf.points, point)), initial_points = None)
            # surfs[surf_id] = adasurf(np.vstack((surf[2], point)), config)
            return True
        else:
            return False

    def remove_poor_support_surface(fail):
        newfail = np.array([]).reshape(0, 3)
        index_to_remove = []
        print '***current dropping threshold is ', config.filter_rate * len(points)
        for (surf, index) in zip(surfs, range(len(surfs))):
            supporting = len(surf.points)
            if supporting < config.filter_rate * len(points): # if this surface is poor supported
                # remove the surf and add its supporting points back to fail
                print "***drop one"
                newfail = np.vstack((newfail, surf.points))
                index_to_remove.append(index)
        for index in sorted(index_to_remove, reverse=True):
            del surfs[index]
        return np.vstack((np.array(fail).reshape((-1, 3)), newfail.reshape((-1, 3))))

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points

    nstd = np.std(npoints[:, 1:-1])
    print 'nstd of all points in this segment', nstd
    print 'len(surf)', len(surfs)
    fail = Pipecycle(npoints, judge_point, new_surf, remove_poor_support_surface)

    fig = None
    if paint_when_end:
        fig = paint_surfs(surfs, npoints, title = title, show = False)

    return surfs, npoints, fail, (fig, )

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')

    import time
    starttime = time.clock()
    surfs, npoints, _, _ = identifysurf(c, AdaSurfConfig())
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
        print s.args # surface args
        print s.residuals # MSE
        print len(s.points)
        # print s[2] # npoints
        print '**************************************'

    print len(surfs)


    paint_surfs(surfs, npoints)
