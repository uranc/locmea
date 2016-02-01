"""
Create attributes for the localization framework
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
import numpy as np
import matplotlib.pyplot as plt


class data_out(object):
    """
    Class of various reconstructions
    ----------
    Attributes
    ----------
    """

    def __init__(self, *args, **kwargs):
        self.electrode_pos = args[0]
        self.options = {'p_vres': 5,
                        'p_jlen': 2,
                        'p_erad': 5,
                        'p_maxd': 55,
                        }
        self.options.update(kwargs)
        print 'Parameter options:', self.options
        if kwargs.get('voxels') is not None:
            self.voxels = kwargs.get('voxels')
        else:
            self.voxels = self.create_voxels(self.electrode_pos,
                                             p_vres=self.options['p_vres'],
                                             el_radius=self.options['p_erad'],
                                             max_depth=self.options['p_maxd'],
                                             p_jlen=self.options['p_jlen'])

    def create_voxels(this, electrode_pos, p_vres=5,
                      el_radius=5, max_depth=55, p_jlen=2):
        """
        Create voxel space w.r.t. -m- electrode pos., 1 voxel larger margin
        -------------
        Arguments
        -------------
        electrode_pos - Vector valued neuron morphology position
        param_res - Resolution in micrometer
        elrad - Electrode radius
        max_depth - max reconstruction depth
        ------------
        Returns
        ------------
        """
        jitter_vector = np.random.random((3, )) * p_jlen   # iCSD Leski
        n_elx = np.unique(electrode_pos[:, 0]).shape[0]
        n_ely = np.unique(electrode_pos[:, 1]).shape[0]
        n_elz = np.unique(electrode_pos[:, 2]).shape[0]
        n_el = n_elx * n_ely * n_elz
        # Check position vector formats
        if n_el != electrode_pos[:, 0].shape[0] or n_el != \
            electrode_pos[:, 1].shape[0] or \
                n_el != electrode_pos[:, 2].shape[0] or n_el <= 0:
            print 'Electrode coordinates are wrong'
        el_normal = [n_elx, n_ely, n_elz].index(1)
        # Find min/max values
        min_el = np.ma.array(np.min(electrode_pos, 0), mask=False)
        max_el = np.ma.array(np.max(electrode_pos, 0), mask=False)
        min_el[el_normal] += el_radius
        max_el[el_normal] += max_depth + 1
        min_el.mask[el_normal] = True    # Handle depth differently
        max_el.mask[el_normal] = True
        min_el -= p_vres
        max_el += p_vres + 1
        min_el.mask[el_normal] = False
        max_el.mask[el_normal] = False
        return (np.mgrid[min_el[0]:max_el[0]:p_vres,
                         min_el[1]:max_el[1]:p_vres,
                         min_el[2]:max_el[2]:p_vres].T
                + jitter_vector).T

    def cmp_fwd_matrix(this, electrode_pos, voxels, p_sigma=0.3):
        """
        Calculate the m-by-n forward matrix given by
        1/(4*pi*sigma)*(1/(d(el_pos-vox_pos)^2)
        ------------
        Arguments
        ------------
        elx, ely, elz - Vector valued electrode positions
        vx, vy, vz - Vector valued voxel positions
        """
        vx, vy, vz = voxels
        elx, ely, elz = electrode_pos.T
        # Flatten
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # Electrode geometry
        n_elx = np.unique(elx).shape[0]
        n_ely = np.unique(ely).shape[0]
        n_elz = np.unique(elz).shape[0]
        n_el = n_elx * n_ely * n_elz
        n_v = vx.shape[0]
        # Check position vector formats
        if n_el != elx.shape[0] or n_el != ely.shape[0] \
                or n_el != elz.shape[0] or n_el <= 0:
            print 'Electrode coordinates are wrong'
        fwd_matrix = np.zeros((n_el, n_v))
        for el in np.arange(0, n_el):
            for v in np.arange(0, n_v):
                fwd_matrix[el, v] = np.sqrt((elx[el] - vx[v]) ** 2 +
                                            (ely[el] - vy[v]) ** 2 +
                                            (elz[el] - vz[v]) ** 2)
        return 1. / (fwd_matrix * (4. * np.pi * p_sigma))

    def cmp_inv_matrix(this, fwd_matrix, depth_norm_matrix, p_lmbda=1e-2):
        """
        Computes regularized inverse matrix in the given method
        (Can be a class later on with multiple methods)
        """
        cov_n = np.eye(fwd_matrix.shape[0])
        inv_matrix = np.dot(np.dot(depth_norm_matrix,
                                   np.transpose(fwd_matrix)),
                            np.linalg.inv(
                            np.dot(np.dot(fwd_matrix, depth_norm_matrix),
                                   np.transpose(
                                       fwd_matrix)) + (p_lmbda ** 2) * cov_n))
        return inv_matrix

    def cmp_weight_matrix(this, fwd_matrix, p_depth=1.):
        """
        Calculate the column(depth) normalization matrix given by
        (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
        """
        def get_neighbors(this, voxels, i, j, k, d):
            """
            Looks for the neighboring voxels within -d-
            hamming distance given a (i,j,k)position
            """
        return np.diag(np.power(np.sum(fwd_matrix ** 2, axis=0),
                                1. / (2 * p_depth)))

    def cmp_resolution_matrix(this, fwd_matrix, inv_matrix):
        """
        Calculate the column(depth) normalization matrix given by
        (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
        """
        return np.dot(inv_matrix, fwd_matrix)

    def visualize_reconstructions(this):
        """
        visualize the reconstructions
        """
    def evaluate_localization(this):
        """
        evaluate the localization
        """
    def generate_figure(self):
        """
        Initialize the figure
        """
        self.fig = plt.figure(figsize=(20, 10))
        ax = self.fig.add_subplot(232)
        ax.legend(prop={'size': 12})
        ax.set_ylabel('controls ')
        ax.set_xlabel('time (s)')
        ax.set_xlim(0, 10)
        ax.set_ylim(-5.5, 5.5)

    def write_log(self):
        """
        write optimization log
        """
    def write_output_data(self):
        """
        write output results, parameters, fig details, etc.
        """
