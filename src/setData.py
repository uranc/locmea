"""
@package setData
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
    Data class loaded from file
    """

    def __init__(self, *args, **kwargs):
        """
        @brief      { constructor_description }
        
        @param      self    The object
        @param      args    The args
        @param      kwargs  The kwargs
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
        
        @param      self       The object
        @param      f_name     The f name
        @param      flag_cell  The flag cell
        
        @return     { description_of_the_return_value }
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
        
        @param      self    The object
        @param      f_name  The f name
        
        @return     { description_of_the_return_value }
        """

    def filter_bpass_data(self):
        """
        filter raw data
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """

    def car_data(self):
        """
        common average reference
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """

    def epoch_data(self):
        """
        threshold to get spike time points (can be overwritten)
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """

    def cmp_cov_sensor(self):
        """
        compute the covariance matrix for sensors
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """

    def cmp_pca_ica(self):
        """
        compute pca, ica, dimensionality reduction
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """
