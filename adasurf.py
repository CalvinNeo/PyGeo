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

from scipy.optimize import leastsq

class AdaSurfConfig:
    def __init__(self):
        self.origin_points = 4

    # 待拟合面的函数，x是变量，p是参数
    def surf_fun(self, x, y, params):
        a, b, c = params
        return -(a*x + b*y + c)

def adasurf(points, config):
    # 计算真实数据和拟合数据之间的误差，p是待拟合的参数，x和y分别是对应的真实数据
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

    # 调用拟合函数，第一个参数是需要拟合的差值函数，第二个是拟合初始值，第三个是传入函数的其他参数
    r = leastsq(residuals, [1, 0.5, 1], args=(x1, y1, z1))

    # 打印结果，r[0]存储的是拟合的结果，r[1]、r[2]代表其他信息
    return r[0], MSE(r[0], points)

def paint_surf(a, b, c, points=None):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-1, 1, 0.05)
    Y = np.arange(-1, 1, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = -(X*a + Y*b + c)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if points != None:
        x1 = points[:, 0]
        y1 = points[:, 1]
        z1 = points[:, 2]
        ax.scatter(x1, y1, z1, c='r')
        pl.show()

def paint_surfs(surfs, points, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.1, 1.1)):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ans in surfs:
        a, b, c = ans[0][0], ans[0][1], ans[0][2]
        X = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/100.0)
        Y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/100.0)
        X, Y = np.meshgrid(X, Y)
        Z = -(X*a + Y*b + c)
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, Z, rstride=15, cstride=15)

    ax.set_zlim(zlim[0], zlim[1])
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]
    ax.scatter(x1, y1, z1, c='r')
    pl.show()

def filterex(iterator, predicate):
    dequeyes = collections.deque()
    dequeno = collections.deque()
    try:
        while True:
            x = it.next()
            if predicate(val):
                dequeyes.append(x)
            else:
                dequeno.append(x)
    except StopIteration:
        pass
    return dequeyes, dequeno

def Pipecycle(iterable, predicate, roundclearup = None):
    fucka = 1
    while len(iterable) > 0:
        fucka += 1
        print len(iterable)
        # if fucka > 10:
        #     return
        fail = []
        for x in iterable:
            val = predicate(x)
            if not val:
                fail.append(x)
        iterable = np.array(fail)
        if roundclearup(iterable):
            return

def identifysurf(points, config):
    def same_surf(surf, point):
        # print abs(point[2]-config.surf_fun(point[0], point[1], surf[0])) , surf[1] * 100
        return abs(point[2]-config.surf_fun(point[0], point[1], surf[0])) <= 0.5

    def new_surf(partial_points):
        all_surf = []
        for circum in combinations(partial_points, config.origin_points):
            all_surf.append(adasurf(np.array(circum), config))
        if len(all_surf) > 0:
            surfs.append(min(all_surf, key=lambda x:x[1]))
            return False
        elif len(partial_points) <= config.origin_points:
            return True

    def judge_point(point):
        for surf in surfs:
            if same_surf(surf, point):
                return True
        return False

    def point_normalize(points):
        points = np.array(points)
        points[:, 0] = points[:, 0] - np.mean(points[:, 0])
        points[:, 1] = points[:, 1] - np.mean(points[:, 1])
        points[:, 2] = points[:, 2] - np.mean(points[:, 2])
        return points

    surfs = []

    npoints = point_normalize(points)
    Pipecycle(npoints, judge_point, new_surf)

    return surfs, npoints

if __name__ == '__main__':
    c = np.loadtxt('1.py', comments='#')
    # ans, r = adasurf(c, AdaSurfConfig())
    # print ans, r, np.mean(c[:, 2]), np.std(c[:, 2])

    surfs, npoints = identifysurf(c, AdaSurfConfig())
    print len(surfs), surfs
    xlim = (np.min(npoints[:, 0]), np.max(npoints[:, 0]))
    ylim = (np.min(npoints[:, 1]), np.max(npoints[:, 1]))
    zlim = (np.min(npoints[:, 2]), np.max(npoints[:, 2]))

    paint_surfs(surfs, npoints, xlim, ylim, zlim)
