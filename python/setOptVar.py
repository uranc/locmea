"""
Create attributes for the localization framework
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
import numpy as np


def create_voxels(elx, ely, elz, p_vres=1, el_radius=5, max_depth=55):
    """
    Create voxel space w.r.t. -m- electrode positions, 1 voxel larger margin
    -------------
    Arguments
    -------------
    elx, ely, elz - Vector valued neuron morphology position
    param_res - Resolution in micrometer
    elrad - Electrode radius
    max_depth - max reconstruction depth
    ------------
    Returns
    ------------
    vx, vy, vz - Vector valued voxel positions
    """
    jitter_vector = np.random.random(3)   # iCSD Leski refererence
    n_elx = np.unique(elx).shape[0]
    n_ely = np.unique(ely).shape[0]
    n_elz = np.unique(elz).shape[0]
    n_el = n_elx*n_ely*n_elz
    # Check position vector formats
    if n_el != elx.shape[0] or n_el != ely.shape[0] \
            or n_el != elz.shape[0] or n_el <= 0:
        print 'Electrode coordinates are wrong'
    # Find the electrode normal
    el_normal = [n_elx, n_ely, n_elz].index(1)
    # Find min/max values
    min_el = np.ma.array(np.min([elx, ely, elz], 1), mask=False)
    max_el = np.ma.array(np.max([elx, ely, elz], 1), mask=False)
    min_el[el_normal] += el_radius
    max_el[el_normal] += max_depth + 1
    min_el.mask[el_normal] = True    # Handle depth differently
    max_el.mask[el_normal] = True
    min_el -= p_vres
    max_el += p_vres + 1
    min_el.mask[el_normal] = False
    max_el.mask[el_normal] = False
    return np.mgrid[min_el[0]:max_el[0]:p_vres,
                    min_el[1]:max_el[1]:p_vres,
                    min_el[2]:max_el[2]:p_vres]


    def cmp_fwd_matrix(elx, ely, elz, vx, vy, vz):
    """
    Calculate the m-by-n forward matrix given by
    1/(4*pi*sigma)*(1/(d(el_pos-vox_pos)^2)
    """
    fwd_matrix = np.array(0)


def cmp_inv_matrix(fwd_matrix, reg_par):
    """
    Computes regularized inverse matrix
    """
    inv_matrix = np.array(0)
    return inv_matrix


def cmp_depth_norm_matrix(fwd_matrix, dpth_par):
    """
    Calculate the column(depth) normalization matrix given by
    (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
    """
    depth_norm_matrix = np.array(0)
    return depth_norm_matrix
