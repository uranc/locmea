"""Module for loading data from various files"""


def load_data_from_text(f_name):
    """Load data from txt files"""
    with open(f_name, 'r+') as f:
        read_data = f.read()
    f.closed
    if read_data:
        print "Data loaded."
