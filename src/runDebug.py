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
t_ind = 0
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
opt = opt_out(data, p_vres=20, p_jlen=0, p_maxd=55)
# wform = opt.optimize_waveform()
# wform = ca.DM(wform).full().flatten()
# plt.plot(wform)
# opt.solve_ipopt_reformulate()
opt.solve_ipopt_multi_measurement()
# opt.xress = opt.xres
# opt.xres = opt.xress[:,:,:,t_ind]

# active set
#opt.solve_ipopt_reformulate_tv()


# constraint functions checks.
#opt.solve_ipopt_reformulate_tv_mmv()
#opt.solve_ipopt_reformulate_tv()


# visualize
opt.generate_figure_morphology()

# waveform optimization
@ca.pycallback
def plot_update(self, f):
    x = f.getOutput("x")
    self.plot_data(x, self.X, self.N, self.timeSlotSizes, self.E0, self.V_W,\
    data = self.data)
    return 0