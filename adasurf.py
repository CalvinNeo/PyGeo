#coding:utf8

import numpy as np, scipy
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.optimize import leastsq

def adasurf(points):
    # 待拟合的函数，x是变量，p是参数
    def fun(x, y, params):
        a, b, c = params
        return -(a*x + b*y + c)

    # 计算真实数据和拟合数据之间的误差，p是待拟合的参数，x和y分别是对应的真实数据
    def residuals(params, x, y, z, regularization = 1.0):
        rt = z - fun(x, y, params)
        rt = np.append(rt, np.sqrt(regularization)*params)
        return rt

    def MSE(params, points):
        e = (points[:,2] - fun(points[:,0], points[:,1], params))
        return np.sqrt(np.dot(e.T, e))

    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]

    # 调用拟合函数，第一个参数是需要拟合的差值函数，第二个是拟合初始值，第三个是传入函数的其他参数
    r = leastsq(residuals, [1, 0.5, 1], args=(x1, y1, z1))

    # 打印结果，r[0]存储的是拟合的结果，r[1]、r[2]代表其他信息
    return r[0], MSE(r[0], points)

def paint_surf(a, b, c, points):
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
    x1 = points[:, 0]
    y1 = points[:, 1]
    z1 = points[:, 2]
    ax.scatter(x1, y1, z1, c='r')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    pl.show()

def detectsurf(points):
    pass

if __name__ == '__main__':
    c = np.loadtxt('1.py', comments='#')
    ans, r = adasurf(c)
    print ans, r, np.mean(c[:, 2]), np.std(c[:, 2])
    # paint_surf(1, 1, 1, c)
    paint_surf(ans[0], ans[1], ans[2], c)
