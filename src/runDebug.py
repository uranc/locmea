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
import time

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
                        't_ind': 30, 't_int': 1, 'sigma': 1e-3, 
                        'flag_depthweighted': True,
                        'flag_parallel': False,
                        'datafile_name': 'output_file'}
opt = opt_out(data, **optimization_options)
# opt.solve_ipopt_multi_measurement_thesis()
opt.solve_ipopt_multi_measurement_slack()


# visualize
# vis = visualize(data=data, loc=loc)
# vis.show_snapshot()