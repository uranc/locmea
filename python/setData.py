"""
Module for loading data from various files
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
import numpy as np


def load_data_from_text(f_name):
    """Load data from txt files"""
    with open(f_name, 'r+') as f:
        read_data = f.read()
    f.closed
    if read_data:
        print "Data loaded."
        print read_data(1)
    return np.array(read_data)


# def high_pass_data():
# def car_data():
# def epoch_data():
