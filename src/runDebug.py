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

# Data path/filename
t_ind = 38
data_path = '../data/'
file_name = data_path + 'data_sim_low.hdf5'

data_options = {'flag_cell': True, 'flag_electode': False}
data = data_in(file_name, **data_options)

# Inverse solution
# localization_options = {'p_vres':20, 'p_jlen':0, 'p_erad': 5, 't_ind': 38}
# loc = data_out(data, **localization_options)

# loc.cmp_sloreta()
# loc.xres = loc.res[:, t_ind]
# ccell = loc.evaluate_localization()
# Optimize
optimization_options = {'p_vres':10, 'p_jlen':0, 'p_erad': 5,
                        'solver': 'ipopt', 'method': 'grad',
                        't_ind': 1, 't_int': 1, 'sigma': 1, 'flag_depthweighted': True}
opt = opt_out(data, **optimization_options)
opt.set_optimization_variables_thesis()
#grd = opt.add_s_smooth_costs_constraints()
#opt.add_s_smooth_costs_constraints()
opt.optimize_waveform()
#opt.add_smoothness_costs_constraints()
#opt.add_smoothness_costs_constraints()
# wform = opt.optimize_waveform()
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
# vis = visualize(data=data, loc=opt)
# vis.show_snapshot()