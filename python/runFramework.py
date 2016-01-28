"""
Inverse optimizer example
"""
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
    data_path = ''
    file_name = data_path + 'data_sim_l8e3.hdf5'
    print "Looking for file" + file_name
    data = data_in(file_name, flag_cell=True, flag_electode=False)
    loc = data_out(data.electrode_pos, p_vres=20, p_jlen=0)


if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
