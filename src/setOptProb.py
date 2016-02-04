"""
Create attributes for the optimization problem
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setInvProb import data_out
import casadi as ca
import numpy as np


class opt_out(data_out):
    """
    Class for the optimization problem
    Child of inverse problem
    """
    def __init__(self, *args, **kwargs):
        data_out.__init__(self, *args, **kwargs)

    @ca.pycallback
    def plot_updates(self, f):
        """
        Gets inter-optimization plots
        """
        x = f.getOutput('x')
        self.plot_data(x)
        return 0

    def cmp_gradient(self):
        """
        cmp_gradient
        """
        vx, vy, vz = self.voxels
        ind = 300
        i, j, k = np.unravel_index(ind, vx.shape)
        ind = np.ravel_multi_index((i, j, k), vx.shape)
