"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

import sys
from setData import load_data_from_text
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
    file_name = data_path + 'CAP.csv'
    print "Looking for file" + file_name
    data_raw = load_data_from_text(file_name)
    print data_raw.shape
    # Electrode positions - 3 vectors
    # voxel_pos = create_voxels(elPos[0], elPos[1], elPos
    # [2])

    # Create MNE-Python Raw data format
    # mne_raw = create_mne_format()


if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
