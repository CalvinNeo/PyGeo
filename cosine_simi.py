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


def paint_surfs(surfs, points, show = True, title = '', fail_points = []):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    xlim = (np.min(points[:, 0]), np.max(points[:, 0]))
    ylim = (np.min(points[:, 1]), np.max(points[:, 1]))
    zlim = (np.min(points[:, 2]), np.max(points[:, 2]))
    for ans, surf_id in zip(surfs, range(len(surfs))):
        a, b, c, d = ans.args[0], ans.args[1], ans.args[2], ans.args[3]
        X = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.0)
        Y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.0)
        X, Y = np.meshgrid(X, Y)
        Z = -(X*a + Y*b + d)
        # s = ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15)
        x1 = ans.points[:, 0]
        y1 = ans.points[:, 1]
        z1 = ans.points[:, 2]
        ax.scatter(x1, y1, z1, c='rcygm'[surf_id % 5], marker='o^sd*+xp'[int(surf_id/5)])

    x1 = fail_points[:, 0]
    y1 = fail_points[:, 1]
    z1 = fail_points[:, 2]
    ax.scatter(x1, y1, z1, c='k', marker='o')

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


class Surface:
    def printf(self):
        print str((self.args, self.residuals, self.initial_points))

    def addpoint(self, p):
        self.points = np.vstack((self.points, p))

    def normalizer(self):
        return math.sqrt(self.args[0]**2 + self.args[1]**2 + 1)

    def vec(self):
        return np.array([self.args[0], self.args[1], 1])

    def normvec(self):
        return self.vec() / self.normalizer()

    def __str__(self):
        return str(tuple(self.args, self.residuals, self.points, self.initial_points))

    def belong_point(self, point):
        # if possible, add one point to one of the surfaces
        A = self.args.reshape(1, 4)
        X = np.array([point[0], point[1], point[2], 1]).reshape(4, 1)
        upper = np.dot(A, X)[0,0]
        lower = math.sqrt(np.dot(A[:, 0:3], (A[:, 0:3]).reshape(3, 1)))
        e = abs(upper / lower)
        return e <= config.pointsame_threshold, e

    def __init__(self, *initial_data, **kwargs):
        self.residuals = 0
        self.points = np.array([]).reshape(0, 3)
        self.args = np.array([1.0, 1.0, 1.0, 1.0])
        self.initial_points = np.array([]).reshape(0, 3)
        self.current_fun = 0

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

class AdaSurfConfig:
    def __init__(self, *initial_data, **kwargs):
        self.origin_points = 5
        self.most_combination_points = 20
        self.same_threshold = 0.2 # the smaller, the more accurate when judging two surfaces are identical, more surfaces can be generated
        self.pointsame_threshold = 1.0

        self.combine_thres = 0.9
        self.filter_count = 30
        self.fail_count = 5

        self.fitting_fun = [self.surf_fun0, self.surf_fun1]

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    # 待拟合面的函数，x是变量，p是参数
    def surf_fun0(self, x, y, params):
        a, b, c, d = params
        return -(a*x + b*y + d)

    def surf_fun1(self, x, z, params):
        a, b, c, d = params
        return -(a*x + c*z + d)

    def surf_fun2(self, x, z, params):
        a, b, c, d = params
        return -(b*x + c*z + d)


    def surf_fun(self, x, y, z, params, fitting_fun):
        if fitting_fun == 0:
            return self.surf_fun0(x, y, params)
        elif fitting_fun == 1:
            return self.surf_fun1(x, z, params)
        else:
            return self.surf_fun2(x, z, params)

    def residuals(self, params, x, y, z, fitting_fun, regularization = 0.0):
        if fitting_fun == 0:
            rt = z - self.surf_fun0(x, y, params)
        elif fitting_fun == 1:
            rt = y - self.surf_fun1(x, z, params)
        else:
            rt = x - self.surf_fun2(y, z, params)
        # rt = np.append(rt, np.sqrt(regularization)*params)
        return rt

ELAPSE_LSQ = 0
ELAPSE_STD = 0

def adasurf(points, config, initial_points = None):
    current_fun = 0

    def residuals(params, x, y, z, regularization = 0.0):
        return config.residuals(params, x, y, z, current_fun, regularization)

    def MSE(params, points):
        e = config.residuals(params, points[:,0], points[:,1], points[:,2], fitting_fun = current_fun)
        return np.sqrt(np.dot(e.T, e)/len(e))

    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]

    r = leastsq(residuals, [1.0, 1.0, 1.0, 1.0], args=(x1, y1, z1))
    
    if r[0][0] > 100 or r[0][1] > 100:
        # overfit
        # print "OVER FIT SWITCH "
        # print "before", r[0]
        current_fun = 1
        r = leastsq(residuals, [1.0, 1.0, 1.0, 1.0], args=(x1, y1, z1))
        if r[0][0] > 100 or r[0][1] > 100:
            current_fun = 2
            r = leastsq(residuals, [1.0, 1.0, 1.0, 1.0], args=(x1, y1, z1))

        # print "after", r[0]

    return Surface(args = r[0], residuals = MSE(r[0], points), points = points, initial_points = initial_points, current_fun = current_fun)

def point_normalize(points):
    points = np.array(points)
    points[:, 0] = points[:, 0] - np.mean(points[:, 0])
    points[:, 1] = points[:, 1] - np.mean(points[:, 1])
    points[:, 2] = points[:, 2] - np.mean(points[:, 2])
    return points

PRINT_COUNT = 0
def identifysurf(points, config, donorm = True, surfs = [], paint_when_end = False, title = '', current_direction = None):
    def same_surf(surf1, surf2):
        v1, v2 = surf1.normvec(), surf2.normvec()
        r = abs(np.dot(v1, v2))
        # print r, v1, v2, surf1.vec(), surf2.vec()
        yes = r > config.combine_thres and abs(surf1.args[3] - surf2.args[3]) < (abs(surf1.args[3]) + abs(surf2.args[3])) / 10.0
        return yes, r

    def combine_surf():
        # combine similar surf, always dealing the last surf
        if len(surfs) > 1:
            for i in xrange(len(surfs) - 1):
                r, e = same_surf(surfs[i], surfs[-1])
                if r:
                    surfs[i].addpoint(surfs[-1].points)
                    print "&&& COMBIMED", e
                    del surfs[-1]
                    break

    def ada_point(unfixed_points):
        MIN_POINT_E = 999999
        MAX_POINT_E = -1
        E = []
        def find_surf(p):
            for surf in surfs:
                r, e = surf.belong_point(p)
                if r:
                    surf.addpoint(p)
                    return True, e
            return False, e
        fail = []
        for p in unfixed_points:
            r, e = find_surf(p)
            MIN_POINT_E = min(e, MIN_POINT_E)
            MAX_POINT_E = max(e, MAX_POINT_E)
            E.append(e)
            if not r:
                fail.append(p)
        print "MIN_POINT_E", MIN_POINT_E
        print "MAX_POINT_E", MAX_POINT_E
        # print np.array([x for x in E if x < config.pointsame_threshold])
        return np.array(fail)                

    def filter_surf(ori_points):
        newfail = np.array([]).reshape(0, 3)
        index_to_remove = []
        for (surf, index) in zip(surfs, range(len(surfs))):
            supporting = len(surf.points)
            print "FILTER:", index, "Supporting", supporting, "config.filter_count", config.filter_count
            if supporting < config.filter_count:
                newfail = np.vstack((newfail, surf.points))
                index_to_remove.append(index)
        for index in sorted(index_to_remove, reverse=True):
            print "***drop one"
            del surfs[index]
        return np.vstack((np.array(ori_points).reshape((-1, 3)), newfail.reshape((-1, 3))))

    def new_surf(partial_points):
        MIN_RESIDUAL = 999999999
        np.random.shuffle(partial_points[:])
        all_surf = []
        len_group = int(math.ceil(len(partial_points)*1.0/config.most_combination_points))
        for group_id in xrange(len_group):
            # print "group_id", group_id
            choices = partial_points[group_id*config.most_combination_points:(group_id+1)*config.most_combination_points, :]
            for circum in combinations(choices, config.origin_points):
                generated_surf = adasurf(np.array(circum), config)
                # global PRINT_COUNT
                # if PRINT_COUNT < 111:
                #     print "sss:", generated_surf.residuals
                #     PRINT_COUNT += 1
                MIN_RESIDUAL = min(generated_surf.residuals, MIN_RESIDUAL)
                if generated_surf.residuals < config.same_threshold:
                    all_surf.append(generated_surf)
                    break
        print "MIN_RESIDUAL", MIN_RESIDUAL
        print "LEN_ALLSURF", len(all_surf)
        if len(all_surf) > 0:
            # choose the best surface
            print "***ADD SURF"
            surfs.append(min(all_surf, key=lambda x:x.residuals))
            return True
        else:
            # can't generate one surface
            return False

    if donorm:
        npoints = point_normalize(points)
    else:
        npoints = points

    rest_points = npoints.copy()
    fail_count = 0
    while len(rest_points) > config.origin_points:
        print "--------------------------", len(rest_points), len(surfs)
        print "LEN_SURFS_BEFORE", len(surfs)
        # a = rest_points.copy()
        # b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        # _, idx = np.unique(b, return_index=True)
        # rest_points = a[idx]
        len_surf_before = len(surfs)
        gen_new_flag = new_surf(rest_points)
        if gen_new_flag:
            print "LEN_SURFS_AFTER", len(surfs)
            rest_points = ada_point(rest_points)
            rest_points = filter_surf(rest_points)
            if len(surfs) <= len_surf_before:
                gen_new_flag = False
            combine_surf()
        if not gen_new_flag:
            if fail_count < config.fail_count:
                print "!!! --- FAIL --- !!!", fail_count
                fail_count += 1
            else:
                print "!!! --- NO FITTED SURF --- !!!"
                break

    print 'len(surf)', len(surfs)
    print "FAILED:", len(rest_points)

    fig = None
    if paint_when_end:
        fig = paint_surfs(surfs, npoints, title = title, show = False)

    return surfs, npoints, rest_points, (fig, )

if __name__ == '__main__':
    c = np.loadtxt('5.py', comments='#')

    config = AdaSurfConfig()
    print 'config', config.__dict__
    import time
    surfs, npoints, fail, _ = identifysurf(c, AdaSurfConfig())
    fail = np.array(fail)
    print 'TOTAL_POINT: ', len(npoints)
    print "----------BELOW ARE SURFACES----------"
    for s,i in zip(surfs, range(len(surfs))):
        print "SURFACE ", i
        print "s.args", s.args # surface args
        print "s.residuals", s.residuals # MSE
        print "len(s.points)", len(s.points)
        # print s[2] # npoints
        print '**************************************'

    print len(surfs)


    paint_surfs(surfs, npoints, fail_points = fail)
