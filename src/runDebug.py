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
t_ind = 0
data_path = '../data/'
file_name = data_path + 'data_sim_low.hdf5'

data_options = {'flag_cell': True, 'flag_electode': False}
data = data_in(file_name, **data_options)

# Inverse solution
localization_options = {'p_vres':20, 'p_jlen':0, 'p_erad': 5}
loc = data_out(data, **localization_options)

loc.cmp_sloreta()
loc.xres = loc.res[:, t_ind]

# Optimize
optimization_options = {'p_vres':20, 'p_jlen':0, 'p_erad': 5,
                        'solver': 'ipopt', 'method': 'grad',
                        't_ind': 30, 't_int': 2, 'sigma': 6}
opt = opt_out(data, **optimization_options)
# wform = opt.optimize_waveform()
# wform = ca.DM(wform).full().flatten()
# plt.plot(wform)
opt.solve_ipopt_multi_measurement()
opt.xress = opt.xres
opt.xres = opt.xress[:,:,:,t_ind]

# active set
#opt.solve_ipopt_reformulate_tv()


# constraint functions checks.
#opt.solve_ipopt_reformulate_tv_mmv()
#opt.solve_ipopt_reformulate_tv()

# visualize
vis = visualize(data=data, loc=opt)
vis.VisualizeSingleFrame()