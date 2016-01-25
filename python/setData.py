"""
Module for loading data from various files
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
import numpy as np
import h5py


class load_data(object):
    """
    Data class loaded from file
    ---------
    Arguments
    ---------
    f_name - filename
    flag_cell - flag cell morphology information  #ASSUME TRUE
    ---------
    Attributes
    ---------
    srate
    flag_pre
    data
    ---------
    Functions
    ---------
    """
    def __init__(self, f_name, flag_cell=True):
        self.f_name = f_name
        self.flag_cell = flag_cell
        print self.f_name
        if flag_cell:
            self.cell, self.data_raw, self.srate = self.load_h5py_data(
                self.f_name, self.flag_cell)
        else:
            self.data_raw, self.srate = self.load_h5py_data(self.f_name,
                                                            self.flag_cell)

    def load_data_from_text(self, f_name):
        """Load data from txt files"""
        with open(f_name, 'r+') as f:
            read_data = f.read()
        f.closed
        return np.array(read_data)

    def load_h5py_data(self, f_name, flag_cell):
        f = h5py.File(self.f_name, 'r')
        print "Cell data avaliable"
        if flag_cell:
            return f['cell'][:], f['electrode'][:], f['srate'][()]
        else:
            return f['data'][:], f['srate'][()]
    # def high_pass_data():
    # def car_data():
    # def epoch_data():
    # def cmp_cov_source():
    # def cmp_cov_meas():
