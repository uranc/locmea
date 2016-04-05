import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
Inverse optimizer example
"""
'''
Copyright (C)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:


class visualize(object):
    """
    Nice class info
    """

    def __init__(self, *args, **kwargs):
        print "Visualization"
        print kwargs.keys()
        for key in kwargs.keys():
        	print key
        #self.data = kwargs[0]
        #self.xres = kwargs[1].xres
        #self.voxels = kwargs[1].voxels
        #self.t_ind = args[1].t_ind
        self.t_ind = 0

    def VisualizeSingleFrame(self):
        """
        Initialize the figure
        """
        data = self.data
        self.fig = plt.figure(figsize=(20, 10))
        cmax = 1e-5
        t_ind = self.t_ind
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        resn = self.xres.reshape(rx.shape)
        resn_ind = np.abs(resn) > cmax
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
               (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
               (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        # csd plot
        ax = self.fig.add_subplot(233)
        self.csdPlot = [ax.plot(data.cell_csd[ind, :].T, label='CSD')[0]]
        ax.set_ylabel('Transmembrane Current(nA)')
        ax.set_xlabel('Time (ms)')
        # second plot
        ax = self.fig.add_subplot(236)
        self.potPlot = [ax.plot(data.electrode_rec.T, '-', label='MEA')[0]]
        ax.set_ylabel('Electrode Potential(mV)')
        ax.set_xlabel('Time (ms)')
        # morphology
        ax = self.fig.add_subplot(131, projection='3d')
        self.morpPlot = []
        # self.morpPlot = [ax.plot(np.r_[data.cell_pos_start[ind, 0],
        #                                data.cell_pos_end[ind, 0]],
        #                          np.r_[data.cell_pos_start[ind, 1],
        #                                data.cell_pos_end[ind, 1]],
        #                          np.r_[data.cell_pos_start[ind, 2],
        #                                data.cell_pos_end[ind, 2]],
        #                          color='k')[0]]
        self.morpPlot.append(ax.scatter(data.electrode_pos[:, 0],
                                        data.electrode_pos[:, 1],
                                        data.electrode_pos[:, 2],
                                        color='b',
                                        marker='.'))  # electrodes
        self.morpPlot.append(ax.scatter(data.cell_pos[ind, 0],
                                        data.cell_pos[ind, 1],
                                        data.cell_pos[ind, 2],
                                        c=data.cell_csd[ind, t_ind],
                                        cmap='RdBu',
                                        marker='o'))  # midpoints
        ax.azim = 165 - 90
        ax.elev = 20
        # second morphology
        ax = self.fig.add_subplot(132, projection='3d')  # , aspect='equal'
        self.recPlot = []
        # self.recPlot = [ax.plot(np.r_[data.cell_pos_start[ind, 0],
        #                               data.cell_pos_end[ind, 0]],
        #                         np.r_[data.cell_pos_start[ind, 1],
        #                               data.cell_pos_end[ind, 1]],
        #                         np.r_[data.cell_pos_start[ind, 2],
        #                               data.cell_pos_end[ind, 2]],
        #                         color='k')]
        self.recPlot.append(ax.scatter(data.electrode_pos[:, 0],
                                       data.electrode_pos[:, 1],
                                       data.electrode_pos[:, 2],
                                       color='b',
                                       marker='.'))  # electrodes
        self.recPlot.append(ax.scatter(rx[resn_ind],
                                       ry[resn_ind],
                                       rz[resn_ind],
                                       c=resn[resn_ind],
                                       cmap='RdBu',
                                       marker='o'))  # midpoints
        ax.azim = 165 - 90
        ax.elev = 20
        # show all
        self.fig.tight_layout()
        plt.show()

    def AnimateActivation(self):
        """
        Initialize the figure
        """
        data = self.data
        self.fig = plt.figure(figsize=(20, 10))
        cmax = 1e-5
        t_ind = self.t_ind
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        resn = self.xres.reshape(rx.shape)
        resn_ind = np.abs(resn) > cmax
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
               (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
               (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        # csd plot
        ax = self.fig.add_subplot(233)
        self.csdPlot = [ax.plot(data.cell_csd[ind, :].T, label='CSD')[0]]
        ax.set_ylabel('Transmembrane Current(nA)')
        ax.set_xlabel('Time (ms)')
        # second plot
        ax = self.fig.add_subplot(236)
        self.potPlot = [ax.plot(data.electrode_rec.T, '-', label='MEA')[0]]
        ax.set_ylabel('Electrode Potential(mV)')
        ax.set_xlabel('Time (ms)')
        # morphology
        ax = self.fig.add_subplot(131, projection='3d')
        self.morpPlot = []
        # self.morpPlot = [ax.plot(np.r_[data.cell_pos_start[ind, 0],
        #                                data.cell_pos_end[ind, 0]],
        #                          np.r_[data.cell_pos_start[ind, 1],
        #                                data.cell_pos_end[ind, 1]],
        #                          np.r_[data.cell_pos_start[ind, 2],
        #                                data.cell_pos_end[ind, 2]],
        #                          color='k')[0]]
        self.morpPlot.append(ax.scatter(data.electrode_pos[:, 0],
                                        data.electrode_pos[:, 1],
                                        data.electrode_pos[:, 2],
                                        color='b',
                                        marker='.'))  # electrodes
        self.morpPlot.append(ax.scatter(data.cell_pos[ind, 0],
                                        data.cell_pos[ind, 1],
                                        data.cell_pos[ind, 2],
                                        c=data.cell_csd[ind, t_ind],
                                        cmap='RdBu',
                                        marker='o'))  # midpoints
        ax.azim = 165 - 90
        ax.elev = 20
        # second morphology
        ax = self.fig.add_subplot(132, projection='3d')  # , aspect='equal'
        self.recPlot = []
        # self.recPlot = [ax.plot(np.r_[data.cell_pos_start[ind, 0],
        #                               data.cell_pos_end[ind, 0]],
        #                         np.r_[data.cell_pos_start[ind, 1],
        #                               data.cell_pos_end[ind, 1]],
        #                         np.r_[data.cell_pos_start[ind, 2],
        #                               data.cell_pos_end[ind, 2]],
        #                         color='k')]
        self.recPlot.append(ax.scatter(data.electrode_pos[:, 0],
                                       data.electrode_pos[:, 1],
                                       data.electrode_pos[:, 2],
                                       color='b',
                                       marker='.'))  # electrodes
        self.recPlot.append(ax.scatter(rx[resn_ind],
                                       ry[resn_ind],
                                       rz[resn_ind],
                                       c=resn[resn_ind],
                                       cmap='RdBu',
                                       marker='o'))  # midpoints
        ax.azim = 165 - 90
        ax.elev = 20
        # show all
        self.fig.tight_layout()
        plt.show()

    def VisualizeForwardMatrix(self, fwd):
        """
        Visualize the exponential decay
        PSF and CTF functions
        """

    def visualize_data(self):
        """
        visualize recordings, epochs, etc..
        frq., power (can be extra function)
        """

    def visualize_cell(self):
        """
        visualize morphology
        frq. power ( can be extra)
        """

    def visualize_cov(self):
        """
        visualize covariance matrix
        """
