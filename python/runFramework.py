"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

import sys
from setData import load_data
# import numpy as np
# from newLFPySimulation import newLFPySimulation


def main():
    """
    Main module for the optimization problem
    Inputs

    data_path
    filename
    electrode_pos  (optional)
    neuron_pos  (optional)
    neuron_csd  (optional)
    ----------
    Attributes
    ----------
    data_raw
    electrode_pos
    voxel_pos
    neuron_pos
    neuron_csd
    reconstructed_pos
    reconstructured_csd
    """

    # Data path/filename
    data_path = ''
    file_name = data_path + 'data_sim_l8e3.hdf5'
    print "Looking for file" + file_name
    loaded_data = load_data(file_name, flag_cell=True)
    print "Data loaded", loaded_data.srate
    # Electrode positions - 3 vectors
    # voxel_pos = create_voxels(elPos[0], elPos[1], elPos
    # [2])


if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
