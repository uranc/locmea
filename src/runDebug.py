"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setData import data_in
from setInvProb import data_out
# from setOptProb import opt_out


# Data path/filename
t_ind = 35
data_path = '../data/'
# file_name = data_path + 'data_sim_l8e3.hdf5'
file_name = data_path + 'data_sim_low.hdf5'  # morphology
print "Looking for file" + file_name
data = data_in(file_name, flag_cell=True, flag_electode=False)
loc = data_out(data, p_vres=20, p_jlen=0, p_erad=10)
loc.cmp_sloreta()
loc.xres = loc.res[:,t_ind]
loc.generate_figure_morphology()

# optimize
# opt = opt_out(data, p_vres=20, p_jlen=0, p_maxd=55)
# opt.solve_ipopt_slack_cost()
# opt.generate_figure()
