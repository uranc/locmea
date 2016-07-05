# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:23:41 2016

@author: dell
"""

import numpy as np
import casadi as ca
import pdb

# Constants
m = 64
n = 1000

# MX Variables
x = ca.MX.sym('x', n, 1)
fw = ca.MX.sym('f', n, 1)
y = ca.MX.sym('y', m, 1)
FWD = ca.MX.sym('FWD', m, n)
fwd = np.random.rand(m, n)
yin = np.random.rand(m, 1)
xin = np.random.rand(n, 1)
x0 = xin
y0 = yin
w = ca.vertcat(x, y)
w0 = ca.vertcat(x0, y0)
lbx = w0 * -100.
ubx = w0 * 100.
lbg = yin * -0.
ubg = yin * 0.
f = 0


def lsq(i):
    return (yin[i] - y[i])

g = []
for i in range(m):
    f += (lsq(i))**2
    g.append(y[i] - ca.dot(fwd[i, :], x))
    #g = ca.vertcat(g,y[i] - ca.dot(fwd[i,:],x))
g = ca.vertcat(*g)
f = ca.norm_2(x)
nlptimes = {'x': w, 'f': f, 'g': g}
solvertimes = ca.nlpsol('solvertimes', 'ipopt', nlptimes)
args = {'x0': w0, 'lbx': lbx, 'ubx': ubx, 'lbg': lbg, 'ubg': ubg}
sol = solvertimes(**args)

# Naive parallel
#simdot = ca.Function('simdot', [x, fw], [ca.dot(fw, x)])
#Y = [yin[i] - simdot(x, FWD[i, :]) for i in range(y.shape[0])]
#fs = 0
# for i in range(y.shape[0]):
#    fs += Y[i]*Y[i]
# pdb.set_trace()
#lsdot = ca.Function('lsdot', [x, FWD], [fs])
#lsdot = lsdot.expand()
#f = lsdot(x, fwd)[0]
# pdb.set_trace()
#nlpdot = {'x': x, 'f': f}
#solverdot = ca.nlpsol('solver', 'ipopt', nlpdot)
# pdb.set_trace()
#result = solver('x0',x0,'lbx',lbx,'ubx',ubx)
# 3 parts
#sim3 = ca.Function('sim3', [x], [ca.mtimes(fwd, x)])
#ls3 = ca.Function('ls3', [x, y], [y - sim3([x])[0]])
