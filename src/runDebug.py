"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setData import data_in
from setInvProb import data_out
from setOptProb import opt_out
from setVisualization import visualize
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle as pc
# Data path/filename
t_ind = 38
data_path = '../data/'
file_name = data_path + 'data_sim_low.hdf5'

data_options = {'flag_cell': True, 'flag_electode': False}
data = data_in(file_name, **data_options)

# Inverse solution
# localization_options = {'p_vres':20, 'p_jlen':0, 'p_erad': 5, 't_ind': 38, 'flag_depthweighted': False}
# loc = data_out(data, **localization_options)

# loc.cmp_sloreta()
# loc.xres = loc.res[:, t_ind]
# ccell = loc.evaluate_localization()
# Optimize
optimization_options = {'p_vres':10, 'p_jlen':0, 'p_erad': 5,
                        'solver': 'ipopt', 'method': 'grad',
                        't_ind': 30, 't_int': 1, 'sigma': 1e-2, 'flag_depthweighted': True}
opt = opt_out(data, **optimization_options)
opt.set_optimization_variables_thesis()
# # opt.cmp_fwd_diff(opt.m)
# opt.add_l1_costs_constraints_thesis()
# opt.add_data_costs_constraints_thesis()
# opt.add_background_costs_constraints_thesis()
opt.add_tv_mask_costs_constraints_thesis()
# # opt.add_s_smooth_costs_constraints_thesis()
# # opt.optimize_waveform()
opt.add_smoothness_costs_constraints_thesis()
# #opt.add_smoothness_costs_constraints()
# # opt.set_optimization_variables_thesis()
# #opt.add_data_costs_constraints_thesis()
# # opt.add_background_costs_constraints_thesis()

# hf,jf = ca.hessian(opt.f, opt.w)
# # jf = ca.jacobian(ca.vertcat(*opt.g), opt.w)
# # # HF = ca.Function('HF', [opt.w], [hf])
# JF = ca.Function('JF', [opt.w], [jf])
# HF = ca.Function('HF', [opt.w], [hf])
# w0 = opt.w(1)
# a = ca.MX.sym('a',1)
# b = ca.MX.sym('b',1)
# c = ca.MX.sym('c',1)
# t = ca.MX.sym('t',1)
# f = a*t*(ca.exp(-t*b)*b*b-ca.exp(-t*c)*c*c)
# x = ca.vertcat(a,b,c)
# hf = ca.jacobian(f, x)
# HF = ca.Function('HF',[x],[hf[0]])

# wform = opt.optimize_waveform(opt.m)
# wform = ca.DM(wform).full().flatten()
# plt.plot(wform)

# slack
#opt.solve_ipopt_multi_measurement()
# opt.xres = opt.res["x"].full()[:opt.x_size*opt.t_int].\
#             reshape((opt.voxels[0, :, :, :].shape[0],
#                      opt.voxels[0, :, :, :].shape[1],
#                      opt.voxels[0, :, :, :].shape[2],
#                      opt.t_int))
# opt.xress = opt.xres
# opt.xres = opt.xress[:,:,:,t_ind]

# 2p
# opt.solve_ipopt_multi_measurement_slack()
# opt.solve_ipopt_multi_measurement_thesis()
# opt.xres_pos = opt.res["x"].full()[:opt.x_size*2*opt.t_int]
# opt.xres_neg = opt.res["x"].full()[opt.x_size*2*opt.t_int:opt.x_size*4*opt.t_int]
# opt.xress = ca.vertcat(opt.xres_pos[:opt.x_size*opt.t_int] - 
# 	                  opt.xres_neg[opt.x_size*opt.t_int:opt.x_size*opt.t_int*2])
# opt.xress = opt.xress.full().reshape((opt.voxels[0, :, :, :].shape[0],
#                      		 opt.voxels[0, :, :, :].shape[1],
#                      		 opt.voxels[0, :, :, :].shape[2],
#                      		 opt.t_int))
# opt.xres = opt.xress[:,:,:,t_ind]


# visualize
# vis = visualize(data=data, loc=loc)
# vis.show_snapshot()