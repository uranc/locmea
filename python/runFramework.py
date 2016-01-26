"""
Inverse optimizer example
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

import sys
from setData import load_data
from setOptVar import (create_voxels, cmp_fwd_matrix, cmp_depth_norm_matrix,
                       cmp_inv_matrix)
import numpy as np
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

    # Optional ground truth
    neuron_pos = loaded_data.cell[:, 0:3]
    neuron_csd = loaded_data.cell[:, 3:]
    electrode_pos = loaded_data.data_raw[:, 0:3]
    # Single optimization
    measurements_y = loaded_data.data_raw[:, 3:]
    # Electrode positions - 3 vectors
    voxel_pos = create_voxels(electrode_pos[:, 0],
                              electrode_pos[:, 1],
                              electrode_pos[:, 2], p_jlen=0, p_vres=20)
    fwd = cmp_fwd_matrix(electrode_pos[:, 0],
                         electrode_pos[:, 1],
                         electrode_pos[:, 2],
                         voxel_pos[0, :].flatten(),
                         voxel_pos[1, :].flatten(),
                         voxel_pos[2, :].flatten())
    dn_fwd = cmp_depth_norm_matrix(fwd)
    inv = cmp_inv_matrix(fwd, dn_fwd, p_lmbda = 1e-3)
    neuron_csd = np.dot(inv,measurements_y)
    print neuron_csd.shape    
    print neuron_csd[:,30].reshape((voxel_pos[0,:].shape))
    print np.max(neuron_csd[:,30].reshape((voxel_pos[0,:].shape)))

if __name__ == "__main__":
    sys.exit(main())
    print "Hallo"
