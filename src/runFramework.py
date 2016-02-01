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
from setInvProb import data_out


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
    file_name = data_path + 'data_sim_l8e3.hdf5'
    print "Looking for file" + file_name
    data = data_in(file_name, flag_cell=True, flag_electode=False)
    loc = data_out(data.electrode_pos, p_vres=20, p_jlen=0)

if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
