"""
Inverse optimizer example
"""
'''
Copyright (C)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

import sys
from setData import data_in
from setOptProb import opt_out


def main():
    """
    Main module for the optimization problem
    ----------
    Inputs
    ----------
    data_path
    filename
    electrode_pos  (optional)
    neuron_pos  (optional)
    neuron_csd  (optional)
    ----------
    Attributes
    ----------
    data_raw
    """

    # Data path/filename
    data_path = '../data/'
    file_name = data_path + 'data_sim_low.hdf5'
    print "Looking for file" + file_name
    data = data_in(file_name, flag_cell=True, flag_electode=False)
    opt = opt_out(data, p_vres=10, p_jlen=0, p_maxd=55)
    opt.minimize_problem()
    resn = opt.res['x'].full().reshape(opt.voxels[0,:,:,:].shape)

if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
