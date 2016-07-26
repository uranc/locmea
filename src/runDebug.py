"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from locData import data_in
from locInverseProblem import data_out
from locOptimizationProblem import opt_out
from locView import visualize
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle as pc
import time
import sys
import getopt


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

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
opts, args = getopt.getopt(sys.argv[1:], "s:h:l:m:n:p:d:")  # sol:hes:norm:sp:dyn:")
for opt, arg in opts:
    if opt in ("-m"):
        p_method = arg 
    elif opt in ("-s"):
        p_solver = arg 
    elif opt in ("-h"):
        p_hessian = arg 
    elif opt in ("-n"):
        p_norm = arg 
    elif opt in ("-p"):
        p_sparse = arg 
    elif opt in ("-d"):
        p_dynamic = arg 
    elif opt in ("-l"):
        p_linsol = arg 

print 'Method is ', p_method
print 'NLP Solver is ', p_solver
print 'Linear Solver is ', p_linsol
print 'Hessian is ', p_hessian
print 'Norm is ', bool(int(p_norm))
print 'Sparsity is ', float(p_sparse)
print 'Dynamics bound is ', float(p_dynamic)

#python -i runDebug.py -m mask -s ipopt -p 0 -d 50 -n 1 -h limited-memory -l mums

# Optimize
optimization_options = {'p_vres':10, 'p_jlen':0, 'p_erad': 10,
                        'solver': p_solver,
                        'hessian': p_hessian,
                        'linsol': p_linsol,
                        'method': p_method,
                        't_ind': 35, 't_int': 30, 
                        'sigma': float(p_sparse),
                        'flag_depthweighted': bool(int(p_norm)),
                        'flag_parallel': False,
                        'datafile_name': 'output_file',
                        'flag_lift_mask': False,
                        'flag_data_mask': True,
                        'flag_callback': True,
                        'flag_write_output': False,
                        'flag_callback_plot': True,
                        'callback_steps': 40,
                        'p_dyn': float(p_dynamic)
                        }

# optimization_options = {'p_vres':10, 'p_jlen':0, 'p_erad': 10,
#                         'solver':' ',
#                         'hessian': 'p_hessian',
#                         'linsol': 'p_linsol',
#                         'method': 'p_method',
#                         't_ind': 35, 't_int': 1, 
#                         'sigma': float(0),
#                         'flag_depthweighted': bool(int(0)),
#                         'flag_parallel': False,
#                         'datafile_name': 'output_file',
#                         'flag_lift_mask': False,
#                         'flag_data_mask': True,
#                         'flag_callback': True,
#                         'flag_callback_plot': True,
#                         'callback_steps': 40,
#                         'p_dyn': float(1)
#                         }


opt = opt_out(data, **optimization_options)
if p_method == 'thesis':
	opt.solve_ipopt_multi_measurement_thesis()
elif p_method == 'mask':
	opt.solve_ipopt_multi_measurement_only_mask()

# 
# opt.solve_ipopt_multi_measurement_slack()
# opt.solve_ipopt_multi_measurement_2p()
# ev = opt.evaluate_localization()
# visualize
# vis = visualize(data=data, loc=opt)
# vis.show_snapshot()
# vis.save_snapshot()
# opt.set_optimization_variables_thesis()
opt.initialize_variables()
# 
# opt.xres = np.array(opt.w(0)['a'].full())
# opt.xreal = opt.get_ground_truth()[0]
# opt.xreal[opt.xreal.nonzero()[0],:]
# vis = visualize(data=data, loc=opt)
# vis.show_snapshot()
# file_name = data_path + 'data_sim_low_.hdf5'

# data_options = {'flag_cell': True, 'flag_electode': False}
# data3 = data_in(file_name, **data_options)
# elec_pos3 = data3.electrode_pos
# pots3 = np.array(data3.electrode_rec[:, 35].flatten().reshape(256,1))
# params = {
#     'gdX': 10,
#     'gdY': 10,
#     'gdZ': 10,
#     'n_sources': 256,
# }
# k2 = KCSD(elec_pos3, pots3, params)
# k2.estimate_pots()
# k2.estimate_csd()
# k2.plot_all()

# from mayavi import mlab
# opt.xreal = opt.get_ground_truth()[0].reshape(opt.voxels[0,:,:,:].shape)
# csd = k.solver.estimated_pots[:,:,:,0]
# # csd = k.solver.estimated_csd[:,:,:,0]
# csd = k2.solver.estimated_pots[:,:,:,0]
# csdreal = opt.xreal
# #setting up a proper gui backend

# mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(csd),
#                         plane_orientation='x_axes',
#                         slice_index=10,
#                         )
# mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(csd),
#                         plane_orientation='y_axes',
#                         slice_index=10,
#                         )
# mlab.outline()

# mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(csdreal),
#                         plane_orientation='x_axes',
#                         slice_index=10,
#                         )
# mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(csdreal),
#                         plane_orientation='y_axes',
#                         slice_index=10,
#                         )
# mlab.outline()