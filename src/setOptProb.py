"""
Create attributes for the optimization problem
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setInvProb import data_out
import casadi as ca
import matplotlib.pyplot as plt


class opt_out(data_out):
    """
    Class for the optimization problem
    Child of inverse problem
    """
    def __init__(self, *args, **kwargs):
        data_out.__init__(self, *args, **kwargs)
        self.fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        self.casadi_fwd = ca.SX.sym("x", self.fwd.shape)

    @ca.pycallback
    def plot_updates(self, f):
        """
        Gets inter-optimization plots
        """
        x = f.getOutput('x')
        self.plot_data(x)
        return 0

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
