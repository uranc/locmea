"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setData import data_in
from setOptProb import opt_out
from setInvProb import data_out
# Data path/filename
data_path = '../data/'
file_name = data_path + 'data_sim_low.hdf5'
print "Looking for file" + file_name
data = data_in(file_name, flag_cell=True, flag_electode=False)
loc = data_out(data, p_vres=20, p_jlen=0)
opt = opt_out(data, p_vres=20, p_jlen=0, p_maxd=45)