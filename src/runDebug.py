"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setData import data_in
from setInvProb import data_out
from setOptProb import opt_out
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
#loc = data_out(data, p_vres=5, p_jlen=0, p_erad=5)
#loc.cmp_sloreta()
#loc.xres = loc.res[:, t_ind]
#loc.generate_figure_morphology()

# optimize
opt = opt_out(data, p_vres=10, p_jlen=0, p_maxd=55)
#wform = opt.optimize_waveform()
#wform = ca.DM(wform).full().flatten()
#plt.plot(wform)
#opt.solve_ipopt_slack_cost()
#opt.solve_ipopt_reformulate()
opt.solve_ipopt_multi_measurement()
#opt.generate_figure_morphology()
#fwd = opt.cmp_fwd_matrix(opt.electrode_pos, opt.voxels)
#dw = opt.cmp_weight_matrix(fwd)
#nn = np.diag(dw).reshape(opt.voxels[0,:,:,:].shape)
#nf = fwd[0,:].reshape(opt.voxels[0,:,:,:].shape)
#plt.imshow(nf[:,0,:])

@ca.pycallback
def plot_update(self, f):
    x = f.getOutput("x")
    self.plot_data(x, self.X, self.N, self.timeSlotSizes, self.E0, self.V_W,\
    data = self.data)
    return 0