"""
Create attributes for the localization framework
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
import numpy as np


def create_voxels(elx, ely, elz, p_vres=1, el_radius=5,
                  max_depth=55, p_jlen=2):
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
    jitter_vector = np.random.random(3) * p_jlen   # iCSD Leski refererence
    n_elx = np.unique(elx).shape[0]
    n_ely = np.unique(ely).shape[0]
    n_elz = np.unique(elz).shape[0]
    n_el = n_elx * n_ely * n_elz
    # Check position vector formats
    if n_el != elx.shape[0] or n_el != ely.shape[0] \
            or n_el != elz.shape[0] or n_el <= 0:
        print 'Electrode coordinates are wrong'
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
    return (np.mgrid[min_el[0]:max_el[0]:p_vres, min_el[1]:max_el[1]:p_vres,
            min_el[2]:max_el[2]:p_vres].T + jitter_vector).T


def cmp_fwd_matrix(elx, ely, elz, vx, vy, vz, p_sigma=0.3):
    """
    Calculate the m-by-n forward matrix given by
    1/(4*pi*sigma)*(1/(d(el_pos-vox_pos)^2)
    ------------
    Arguments
    ------------
    elx, ely, elz - Vector valued electrode positions
    vx, vy, vz - Vector valued voxel positions
    """
    n_elx = np.unique(elx).shape[0]
    n_ely = np.unique(ely).shape[0]
    n_elz = np.unique(elz).shape[0]
    n_el = n_elx * n_ely * n_elz
    n_vx = np.unique(vx).shape[0]
    n_vy = np.unique(vy).shape[0]
    n_vz = np.unique(vz).shape[0]
    n_v = n_vx * n_vy * n_vz
    # Check position vector formats
    if n_el != elx.shape[0] or n_el != ely.shape[0] \
            or n_el != elz.shape[0] or n_el <= 0:
        print 'Electrode coordinates are wrong'
    fwd_matrix = np.zeros((n_el, n_v))
    for el in np.arange(0, n_el):
        for v in np.arange(0, n_v):
            fwd_matrix[el, v] = np.sqrt((elx[el] - vx[v])**2 +
                                        (ely[el] - vy[v])**2 +
                                        (elz[el] - vz[v])**2)
    return 1./(fwd_matrix*(4.*np.pi*p_sigma))


def cmp_inv_matrix(fwd_matrix, depth_norm_matrix, method='sLoreta'):
    """
    Computes regularized inverse matrix in the given method
    (Can be a class later on with multiple methods)
    """
    inv_matrix = np.array(0)
    return inv_matrix


def cmp_depth_norm_matrix(fwd_matrix, p_depth=1.):
    """
    Calculate the column(depth) normalization matrix given by
    (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
    """
    return np.diag(np.power(np.sum(fwd_matrix**2, axis=0), 1./(2*p_depth)))
