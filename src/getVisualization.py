"""
Visualize attributes for the localization framework
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class visualize_this(object):
    """
    Class with visualization functions
    -------
    Attributes
    -------
    """
    def __init__(self, *args, **kwargs):
        print "Ingredients perhaps"

    def visualize_setup(cell, electrode, voxels):
        """
        Visualize electrode positions, cell, and the voxels
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        elx, ely, elz = electrode
        vx, vy, vz = voxels
        cx, cy, cz = cell
        ax.scatter(elx, ely, elz, c='r', marker='o')
        ax.scatter(vx, vy, vz, c='r', marker='.')
        ax.scatter(cx, cy, cz, c='r', marker='*')
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        plt.show()

    def visualize_localization(cell, electrode, rec):
        """
        Visualize source localization
        """

    def visualize_fwd_matrix(fwd):
        """
        Visualize the exponential decay
        PSF and CTF functions
        """
