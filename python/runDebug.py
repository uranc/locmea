"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

from setData import data_in
from setOptProb import opt_out


# Data path/filename
data_path = ''
file_name = data_path + 'data_sim_l8e3.hdf5'
print "Looking for file" + file_name
data = data_in(file_name, flag_cell=True, flag_electode=False)
opt = opt_out(data.electrode_pos, p_vres=20, p_jlen=0)
