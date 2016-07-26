"""
@package locData
@author Cem Uran <cemuran@gmail.com> 
Copyright (C) This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.
"""
import h5py
import pickle as pc


class data_in(object):
    """
    Data class loaded from file or LFPy
    """

    def __init__(self, *args, **kwargs):
        """
        @brief      { Data object }
        
        @param      self    The data object
        @param      args    The arguments
        @param      kwargs  The arguments
        """
        self.f_name = args[0]
        self.flag_cell = kwargs.get('flag_cell')
        print args, kwargs
        print self.f_name
        print self.flag_cell
        if self.flag_cell:
            (self.cell_pos_start, self.cell_pos, self.cell_pos_end,
             self.cell_csd, self.electrode_pos, self.electrode_rec,
             self.srate) = self.load_h5py_data(self.f_name, self.flag_cell)
        else:
            self.data_raw, self.srate = self.load_h5py_data(self.f_name,
                                                            self.flag_cell)

    def load_h5py_data(self, f_name, flag_cell):
        """
        load h5py data optionally cell and electrode positions
        
        @param      self       Data object
        @param      f_name     Filename
        @param      flag_cell  Cell data avaliable? - Binary Flag
        
        @return     { Returns data }
        """
        f = h5py.File(self.f_name, 'r')
        print "Cell data avaliable"
        if flag_cell:
            return (f['cell'][:, 0:3], f['cell'][:, 3:6], f['cell'][:, 6:9],
                    f['cell'][:, 9:], f['electrode'][:, 0:3],
                    f['electrode'][:, 3:], f['srate'][()])
        else:
            return f['data'][:], f['srate'][()]

    def load_with_pickle(self, f_name):
        """
        load h5py data optionally cell and electrode positions
        
        @param      self    Data object
        @param      f_name  Filename
        
        @return     { None }
        """

    def filter_bpass_data(self):
        """
        filter raw data
        
        @param      self  Data object
        
        @return     { None }
        """

    def car_data(self):
        """
        common average reference
        
        @param      self  Data object
        
        @return     { None }
        """

    def epoch_data(self):
        """
        threshold to get spike time points (can be overwritten)
        
        @param      self  Data object
        
        @return     { None }
        """

    def cmp_cov_sensor(self):
        """
        compute the covariance matrix for sensors
        
        @param      self  Data object
        
        @return     { None }
        """

    def cmp_pca_ica(self):
        """
        compute pca, ica, dimensionality reduction
        
        @param      self  Data object
        
        @return     { None }
        """
