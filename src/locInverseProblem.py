"""
@package locInverseProblem
@author Cem Uran <cemuran@gmail.com> 
Copyright (C) This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.
"""
import numpy as np
from scipy.spatial.distance import cdist
import pickle as pc
import os, os.path

class data_out(object):
    """
    Class of various reconstructions
    """
    def __init__(self, *args, **kwargs):
        self.data = args[0]    # does not have to be data dependant
        self.electrode_pos = self.data.electrode_pos
        self.options = {'p_vres': 5,
                        'p_jlen': 2,
                        'p_erad': 5,
                        'p_maxd': 55,
                        't_ind': 0,
                        }
        self.options.update(kwargs)
        self.t_ind = self.options['t_ind']
        print 'Parameter options:', self.options
        if kwargs.get('voxels') is not None:
            self.voxels = kwargs.get('voxels')
        else:
            self.voxels = self.create_voxels(self.electrode_pos,
                                             p_vres=self.options['p_vres'],
                                             el_radius=self.options['p_erad'],
                                             max_depth=self.options['p_maxd'],
                                             p_jlen=self.options['p_jlen'])

    def cmp_sloreta(self):
        """
        computes sloreta inverse solution
        
        @param      self  The inverse object
        
        @return     { None }
        """
        self.fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        self.dw = self.cmp_weight_matrix(self.fwd)
        self.inv = self.cmp_inv_matrix(self.fwd, self.dw)
        self.res = np.dot(self.inv, self.data.electrode_rec)

    def create_voxels(self, electrode_pos, p_vres=5,
                      el_radius=5, max_depth=55, p_jlen=2, flag_bound=False):
        """
        Create voxel space w.r.t. electrode positions
        
        @param      self           The inverse problem object
        @param      electrode_pos  Electrode positions
        @param      p_vres         Voxel resolution
        @param      el_radius      Interelectrode distance
        @param      max_depth      Maximum distance from the electrodes
        @param      p_jlen         Jitter value for the voxels
        @param      flag_bound     Sets Voxel Volume larger then the electrode area
        
        @return     { Voxels in as numpy array }
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
        elec_normal = [n_elx, n_ely, n_elz].index(1)
        # Find min/max values
        min_el = np.ma.array(np.min(electrode_pos, 0), mask=False)
        max_el = np.ma.array(np.max(electrode_pos, 0), mask=False)
        min_el[elec_normal] += el_radius
        max_el[elec_normal] += max_depth + 1
        min_el.mask[elec_normal] = True    # Handle depth differently
        max_el.mask[elec_normal] = True
        max_el += p_vres
        if flag_bound:
            min_el -= p_vres
            max_el += 1
        min_el.mask[elec_normal] = False
        max_el.mask[elec_normal] = False
        return (np.mgrid[min_el[0]:max_el[0]:p_vres,
                         min_el[1]:max_el[1]:p_vres,
                         min_el[2]:max_el[2]:p_vres].T
                + jitter_vector).T

    def cmp_fwd_matrix(self, electrode_pos, voxels, p_sigma=0.3):
        """
        Calculate the m-by-n forward matrix given by
        1/(4*pi*sigma)*(1/(d(el_pos-vox_pos)^2)
        
        @param      self           The inverse problem object
        @param      electrode_pos  The electrode position
        @param      voxels         The voxels
        @param      p_sigma        Extracellular conductance
        
        @return     { description_of_the_return_value }
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

    def cmp_inv_matrix(self, fwd_matrix, depth_norm_matrix,
                       p_lmbda=1e-2, snr=5):
        """
        Computes regularized inverse matrix in the given method 
        
        @param      self               The inverse problem object
        @param      fwd_matrix         The forward matrix
        @param      depth_norm_matrix  The depth normalize matrix
        @param      p_lmbda            Regularization parameter
        @param      snr                Signal-to-Noise ratio
        
        @return     { description_of_the_return_value }
        """
        cov_n = np.eye(fwd_matrix.shape[0])
        p_lmbda = np.trace(np.dot(np.dot(fwd_matrix, depth_norm_matrix),
                                  fwd_matrix.T))/(np.trace(cov_n)*snr**2)
        inv_matrix = np.dot(np.dot(depth_norm_matrix,
                                   np.transpose(fwd_matrix)),
                            np.linalg.inv(
                            np.dot(np.dot(fwd_matrix, depth_norm_matrix),
                                   np.transpose(
                                       fwd_matrix)) + (p_lmbda ** 2) * cov_n))
        return inv_matrix

    def cmp_weight_matrix(self, fwd_matrix, p_depth=1.):
        """
        Calculate the column(depth) normalization matrix given by
        (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
        
        @param      self        The inverse object
        @param      fwd_matrix  The forward matrix
        @param      p_depth     Depth normalization norm order
        
        @return     { description_of_the_return_value }
        """
        def get_neighbors(self, voxels, i, j, k, d):
            """
            Looks for the neighboring voxels within -d-
            hamming distance given a (i,j,k)position
            """
        return np.diag(np.power(np.sum(fwd_matrix ** 2, axis=0),
                                1. / (2 * p_depth)))

    def cmp_resolution_matrix(self, fwd_matrix, inv_matrix):
        """
        Calculate the column(depth) normalization matrix given by
        (1./sum(a_i^2))^depth_par - column norm for fwd_matrix[:,i]
        
        @param      self        The inverse object
        @param      fwd_matrix  The forward matrix
        @param      inv_matrix  The inverse matrix
        
        @return     { Resolution Matrix }
        """
        return np.dot(inv_matrix, fwd_matrix)

    def write_with_pickle(self, data_to_save):
        """
        @brief      { writes the results in a file }
        
        @param      self          The inverse object.
        @param      data_to_save  The data to save
        
        @return     { None }
        """
        fname = '../results/'+self.datafile_name + \
                    '/' + self.datafile_name 
        # mkdir
        if not os.path.exists(os.path.dirname(fname)):
            try:
                os.makedirs(os.path.dirname(fname))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        print fname + ' written.'
        with open(fname, 'wb') as f:
            pc.dump(data_to_save, f)

    def load_with_pickle(self, fname):
        """
        @brief      Loads with pickle.
        
        @param      self   The inverse object.
        @param      fname  The filename
        
        @return     { Loaded file }
        """
        return pc.load(file(fname))

    def evaluate_localization(self):
        """
        evaluate the localization
        
        @param      self  The inverse object
        
        @return     { Evaluation based on closest charges }
        """
        # if not self.res.any():
        #     "You don't have a reconstruction!"
        # if not self.data.cell_pos.any():
        #     "You don't have a ground truth (data.cell_pos)!"
        data = self.data
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind_cell = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
                   (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
                   (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        vis_cell_pos = data.cell_pos[ind_cell.nonzero()[0],:]
        ind_rec = self.xres.nonzero()[0]
        rec_pos = self.voxels.reshape(3,-1).T[ind_rec,:]
        closest_ind_to_cell = np.argmin(cdist(vis_cell_pos,rec_pos),1)
        closest_ind_to_rec = np.argmin(cdist(vis_cell_pos,rec_pos),0)
        return self.xres[closest_ind_to_cell]-data.cell_csd[ind_cell,self.t_ind]