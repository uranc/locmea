"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setData import data_in
from setInvProb import data_out
# from setOptProb import opt_out
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
# Data path/filename
t_ind = 35
data_path = '../data/'
# file_name = data_path + 'data_sim_l8e3.hdf5'
file_name = data_path + 'data_sim_low.hdf5'  # morphology
print "Looking for file" + file_name
data = data_in(file_name, flag_cell=True, flag_electode=False)
loc = data_out(data, p_vres=20, p_jlen=0, p_erad=10)
loc.cmp_sloreta()
loc.xres = loc.res[:, t_ind]
# loc.generate_figure_morphology()

# optimize
# opt = opt_out(data, p_vres=20, p_jlen=0, p_maxd=55)
# opt.solve_ipopt_slack_cost()
# opt.generate_figure()

# root finding

cd = ca.SX(data.cell_csd[0, 28:-1])
tlin = ca.SX(np.linspace(0, (cd.shape[0]-1)/data.srate, cd.shape[0]))
t = ca.SX.sym('t')
p1 = ca.SX.sym('p1')
p2 = ca.SX.sym('p2')
p3 = ca.SX.sym('p3')
t1 = ca.SX.sym('t1')
t2 = ca.SX.sym('t2')
t3 = ca.SX.sym('t3')
p = ca.vertcat([p1, p2, p3])
tau = ca.vertcat([t1, t2, t3])
r = ca.vertcat([p1, p2, p3, t1, t2, t3])
f = p1*ca.exp(-t*t1)*t*t1+p2*ca.exp(-t*t2)*t*t2+p3*ca.exp(-t*t3)*t*t3
F = ca.Function('F', [r, t], [f])

Y = [ca.norm_1(cd[i]-F([r, tlin[i]])[0]) for i in range(cd.shape[0])]     
r0 = [1, 0, 0, 1e3, 2e3, 8e3]
nlp_root = {'x': r, 'f': sum(Y)}
root_solver = ca.nlpsol("solver", "ipopt", nlp_root)
args = {}
args["x0"] = r0
args["lbx"] = ca.vertcat([0, -50, -50, 1e2, 1e2, 1e2])
args["ubx"] = ca.vertcat([50, 0, 0, 1e4, 1e4, 1e4])
res = root_solver(args)
rx = res['x']
ry = [(cd[i]-F([rx, tlin[i]])[0]) *
     (cd[i]-F([rx, tlin[i]])[0]) for i in range(cd.shape[0])]
fit = ca.DM(ry).full().flatten()
org = ca.DM(cd).full().flatten()
plt.plot(fit)
plt.plot(org)
plt.show()