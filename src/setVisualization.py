import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import Normalize
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


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class visualize(object):
    """
    Nice class info
    """

    def __init__(self, *args, **kwargs):
        print "Visualization of:"
        for key in kwargs.keys():
        	print key
        self.data = kwargs['data']
        self.xres = kwargs['loc'].xres[:, 0]
        self.datafile_name = kwargs['loc'].datafile_name
        print kwargs['loc'].method
        if kwargs['loc'].method == 'thesis':
            self.sres = kwargs['loc'].sres
        self.voxels = kwargs['loc'].voxels
        #self.t_ind = args[1].t_ind
        self.t_ind = 0
        self.norm = MidpointNormalize(midpoint=0)

    def save_snapshot(self, cmax = 1e-3, t_ind = 35):
        """
        Initialize the figure
        """
        fname = '../results/'+self.datafile_name + \
                    '/' + self.datafile_name + '_final.png'
        data = self.data
        # self.fig = plt.figure(figsize=(20, 10))
        plt.figure(figsize=(20, 10))
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        n_depth = self.voxels.shape[2]
        cs_width = n_depth/2
        resn = self.xres.reshape(rx.shape)
        resn_ind = np.abs(resn) > cmax
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
               (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
               (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        # csd plot
        sss = np.zeros(resn.shape)
        sss[resn_ind] = resn[resn_ind]
        # res_min = np.min(sss)
        # res_max = np.max(sss)
        # res_zero = 1 - res_max/(res_max + np.abs(res_min))
        # orgcmap = mcm.RdBu
        # shiftedcmap = self.shiftedColorMap(orgcmap, midpoint=res_zero, name='shifted')
        for dl in range(n_depth):
            ax1 = plt.subplot2grid((2,n_depth+cs_width*2),(0,dl+cs_width*2))  #  (2,n_depth+2,3+dl)
            ax1.imshow(sss[:, dl, :].T, norm=self.norm, cmap=plt.cm.RdBu, interpolation='none', origin='lower')
            # ax1.set_ylabel('Transmembrane (nA)')
            # ax1.set_xlabel('Time (ms)')
            # second plot
            ax2 = plt.subplot2grid((2,n_depth+cs_width*2),(1,dl+cs_width*2))
            ax2.imshow(sss[:, dl, :].T, norm=self.norm, cmap=plt.cm.RdBu, interpolation='none', origin='lower')
            # ax2.set_ylabel('Electrode Potential(mV)')
            # ax2.set_xlabel('Time (ms)')
        # morphology
        ax = plt.subplot2grid((2,n_depth+cs_width*2),(0,0), colspan=cs_width, rowspan=cs_width, projection='3d')
        ax.scatter(data.electrode_pos[:, 0],
                   data.electrode_pos[:, 1],
                   data.electrode_pos[:, 2],
                   color='b',
                   marker='.')  # electrodes
        ax.scatter(data.cell_pos[ind, 0],
                   data.cell_pos[ind, 1],
                   data.cell_pos[ind, 2],
                   c=data.cell_csd[ind, t_ind],
                   norm=self.norm,
                   cmap='RdBu',
                   marker='o')  # midpoints
        ax.azim = 10
        ax.elev = 7
        # second morphology
        ax = plt.subplot2grid((2,n_depth+cs_width*2),(0,cs_width), colspan=cs_width, rowspan=cs_width, projection='3d')
        ax.scatter(data.electrode_pos[:, 0],
                   data.electrode_pos[:, 1],
                   data.electrode_pos[:, 2],
                   color='b',
                   marker='.')  # electrodes
        ax.scatter(rx[resn_ind],
                   ry[resn_ind],
                   rz[resn_ind],
                   c=resn[resn_ind],
                   norm=self.norm,
                   cmap='RdBu',
                   marker='o')  # midpoints
        ax.azim = 10
        ax.elev = 7
        # show all
        # # self.fig.tight_layout()
        plt.savefig(fname)

    def show_snapshot(self, cmax = 1e-3, t_ind = 35):
        """
        Initialize the figure
        """
        data = self.data
        # self.fig = plt.figure(figsize=(20, 10))
        plt.figure(figsize=(20, 10))
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        n_depth = self.voxels.shape[2]
        cs_width = n_depth/2
        resn = self.xres.reshape(rx.shape)
        resn_ind = np.abs(resn) > cmax
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
               (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
               (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        # csd plot
        sss = np.zeros(resn.shape)
        sss[resn_ind] = resn[resn_ind]
        # res_min = np.min(sss)
        # res_max = np.max(sss)
        # res_zero = 1 - res_max/(res_max + np.abs(res_min))
        # orgcmap = mcm.RdBu
        # shiftedcmap = self.shiftedColorMap(orgcmap, midpoint=res_zero, name='shifted')
        for dl in range(n_depth):
            ax1 = plt.subplot2grid((2,n_depth+cs_width*2),(0,dl+cs_width*2))  #  (2,n_depth+2,3+dl)
            ax1.imshow(sss[:, dl, :].T, norm=self.norm, cmap=plt.cm.RdBu, interpolation='none', origin='lower')
            # ax1.set_ylabel('Transmembrane (nA)')
            # ax1.set_xlabel('Time (ms)')
            # second plot
            ax2 = plt.subplot2grid((2,n_depth+cs_width*2),(1,dl+cs_width*2))
            ax2.imshow(sss[:, dl, :].T, norm=self.norm, cmap=plt.cm.RdBu, interpolation='none', origin='lower')
            # ax2.set_ylabel('Electrode Potential(mV)')
            # ax2.set_xlabel('Time (ms)')
        # morphology
        ax = plt.subplot2grid((2,n_depth+cs_width*2),(0,0), colspan=cs_width, rowspan=cs_width, projection='3d')
        ax.scatter(data.electrode_pos[:, 0],
                   data.electrode_pos[:, 1],
                   data.electrode_pos[:, 2],
                   color='b',
                   marker='.')  # electrodes
        ax.scatter(data.cell_pos[ind, 0],
                   data.cell_pos[ind, 1],
                   data.cell_pos[ind, 2],
                   c=data.cell_csd[ind, t_ind],
                   norm=self.norm,
                   cmap='RdBu',
                   marker='o')  # midpoints
        ax.azim = 10
        ax.elev = 7
        # second morphology
        ax = plt.subplot2grid((2,n_depth+cs_width*2),(0,cs_width), colspan=cs_width, rowspan=cs_width, projection='3d')
        ax.scatter(data.electrode_pos[:, 0],
                   data.electrode_pos[:, 1],
                   data.electrode_pos[:, 2],
                   color='b',
                   marker='.')  # electrodes
        ax.scatter(rx[resn_ind],
                   ry[resn_ind],
                   rz[resn_ind],
                   c=resn[resn_ind],
                   norm=self.norm,
                   cmap='RdBu',
                   marker='o')  # midpoints
        ax.azim = 10
        ax.elev = 7
        # show all
        # # self.fig.tight_layout()
        plt.show()

    def show_movie(self):
        """
        Initialize the figure
        """


    def show_s_field(self):
        """
        @brief      { function_description }
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """
        """
        Initialize the figure
        """
        data = self.data
        self.fig = plt.figure(figsize=(20, 10))
        cmax = 1
        # xres = self.xres
        t_ind = self.t_ind
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        # result
        ress = self.sres.reshape(rx.shape[0],rx.shape[1],rx.shape[2],3)
        resn = self.xres.reshape(rx.shape)
        resn_ind = np.abs(resn) > cmax
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
               (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
               (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        # csd plot
        # morphology
        ax = self.fig.add_subplot(131, projection='3d')
        self.morpPlot = []
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
        # third morphology
        ax = self.fig.add_subplot(133, projection='3d')  # , aspect='equal'
        self.recPlot = []
        self.recPlot.append(ax.scatter(data.electrode_pos[:, 0],
                                      data.electrode_pos[:, 1],
                                      data.electrode_pos[:, 2],
                                      color='b',
                                      marker='.'))  # electrodes
        self.recPlot.append(ax.quiver(rx[resn_ind],
                                      ry[resn_ind],
                                      rz[resn_ind],
                                      ress[resn_ind, 0],
                                      ress[resn_ind, 1],
                                      ress[resn_ind, 2],
                                      cmap='RdBu',
                                      length=5,
                                      pivot='tail',
                                      ))  # midpoints
        ax.azim = 165 - 90
        ax.elev = 20
        # show all
        self.fig.tight_layout()
        plt.show()


    def show_forward_matrix(self, fwd):
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
