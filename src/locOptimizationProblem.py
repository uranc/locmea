"""
@author Cem Uran <cemuran@gmail.com> 
Copyright (C) This program is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.
"""
from locInverseProblem import data_out
import casadi as ca
import numpy as np
from casadi.tools import struct_symMX, entry, repeated
from casadi.tools import *
from scipy.spatial.distance import cdist
import pickle as pc
import os
import os.path
import time
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError:
    raise ImportError('Matplotlib not found. Might cause problem.')


class opt_out(data_out):
    """
    @brief  Class for the optimization problem Child of inverse problem.
    """
    class MyCallback(ca.Callback):
        """
        Callback class gets results during optimization. This is necessary for
        saving intermediate results.
        """
        class MidpointNormalize(Normalize):
            """
            Normalize plot
            """

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                """
                Requires a midpoint
                """
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                """
                Call for the function

                @param      self   The object
                @param      value  The value
                @param      clip   The clipping value

                @return     { Colormap }
                """
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        def __init__(self, name, nx, ng, np, opts={}):
            """
            Callback class is used for the optimization
            
            @param      self  The object
            @param      name  The name
            @param      nx    { Number of optimization variables x }
            @param      ng    { Number of optimization constraints g }
            @param      np    { Number of optimization parameters p }
            @param      opts  Options for plotting and saving intermediate
                              results.
            """
            ca.Callback.__init__(self)
            # self.norm = opt_out.MyCallback.MidpointNormalize(midpoint=0)
            self.norm = opt_out.MyCallback.MidpointNormalize(vmin=-5,vmax =5,midpoint=0)
            self.iter = 0
            self.data_to_save = []
            self.datafile_name = opts['filename']
            self.flag_callback_plot = opts['flag_callback_plot']
            self.flag_callback_output = opts['flag_callback_output']
            self.s_shape = opts['str_shape']
            self.data = opts['data_cb']
            self.voxels = opts['voxels_cb']
            self.p_method = opts['p_method']
            self.gt = opts['gt']
            opts['filename'] = None
            opts['p_method'] = None
            opts['flag_callback_plot'] = None
            opts['flag_callback_output'] = None
            opts['str_shape'] = None
            opts['data_cb'] = None
            opts['voxels_cb'] = None
            opts['gt'] = None
            self.nx = nx
            self.ng = ng
            self.np = np
            opts['input_scheme'] = ca.nlpsol_out()
            opts['output_scheme'] = ['ret']
            self.construct(name, opts)

        def get_n_in(self): return ca.nlpsol_n_out()

        def get_n_out(self): return 1

        def get_sparsity_in(self, i):
            n = ca.nlpsol_out(i)
            if n == 'f':
                return ca.Sparsity. scalar()
            elif n in ('x', 'lam_x'):
                return ca.Sparsity.dense(self.nx)
            elif n in ('g', 'lam_g'):
                return ca.Sparsity.dense(self.ng)
            else:
                return ca.Sparsity(0, 0)

        def save_snapshot(self, xres_mid, fname, cmax=1e-3, t_ind=0):
            """
            Save a figure showing the intermediate results.

            @param      self      The object
            @param      xres_mid  The soultion x.
            @param      fname     The filename to save.
            @param      cmax      The threshold for visualization.
            @param      t_ind     The time point to visualize.

            @return     { None }
            """
            print fname
            fname = fname + '.png'
            data = self.data
            # self.fig = plt.figure(figsize=(20, 10))
            plt.figure(figsize=(20, 10))
            # mask reconstruction volume
            vx, vy, vz = self.voxels
            rx, ry, rz = vx, vy, vz
            vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
            # result
            n_depth = self.voxels.shape[2]
            cs_width = n_depth / 2
            resn = xres_mid[:, t_ind].reshape(rx.shape)
            resReal = self.gt[:,t_ind].reshape(rx.shape)
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
            for dl in range(n_depth):
                # (2,n_depth+2,3+dl)
                ax1 = plt.subplot2grid(
                    (2, n_depth + cs_width * 2), (0, dl + cs_width * 2))
                ax1.imshow(sss[:, dl, :].T, norm=self.norm,
                           cmap=plt.cm.RdBu, interpolation='none', origin='lower')
                # ax1.set_ylabel('Transmembrane (nA)')
                # ax1.set_xlabel('Time (ms)')
                # second plot
                ax2 = plt.subplot2grid(
                    (2, n_depth + cs_width * 2), (1, dl + cs_width * 2))
                ax2.imshow(resReal[:, dl, :].T, norm=self.norm,
                           cmap=plt.cm.RdBu, interpolation='none', origin='lower')
                # ax2.set_ylabel('Electrode Potential(mV)')
                # ax2.set_xlabel('Time (ms)')
            # morphology
            ax = plt.subplot2grid((2, n_depth + cs_width * 2), (0, 0),
                                  colspan=cs_width, rowspan=cs_width, projection='3d')
            ax.scatter(data.electrode_pos[:, 0],
                       data.electrode_pos[:, 1],
                       data.electrode_pos[:, 2],
                       color='b',
                       marker='.')  # electrodes
            ax.scatter(data.cell_pos[ind, 0],
                       data.cell_pos[ind, 1],
                       data.cell_pos[ind, 2],
                       c=data.cell_csd[ind, 38],
                       norm=self.norm,
                       cmap='RdBu',
                       marker='o')  # midpoints
            ax.azim = 10
            ax.elev = 7
            # second morphology
            ax = plt.subplot2grid((2, n_depth + cs_width * 2), (0, cs_width),
                                  colspan=cs_width, rowspan=cs_width, projection='3d')
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

        def eval(self, arg):
            """
            @brief      { This is the function that CasADi calls at every step.
                        }
            
            @param      self  The object
            @param      arg   The argument
            
            @return     { 0 }
            """
            darg = {}
            for (i, s) in enumerate(ca.nlpsol_out()):
                darg[s] = arg[i]
            sol = darg['x']
            # print self.str_shape
            self.mid_res = self.s_shape(sol)
            if self.p_method == 'thesis':
                xres_mid = self.mid_res['a'].full()
            elif self.p_method == 'mask':
                xres_mid = self.mid_res['a'].full()
            elif self.p_method == 'slack':
                xres_mid = self.mid_res['x'].full()
            elif self.p_method == '2p':
                xres_mid = self.mid_res['xs_pos'].full() - self.mid_res['xs_neg'].full()
            print xres_mid.shape
            fname = '../results/' + self.datafile_name + \
                    '/' + 'iter_' + '{:04d}'.format(self.iter)
            if self.flag_callback_output:
                if not os.path.exists(os.path.dirname(fname)):
                    try:
                        os.makedirs(os.path.dirname(fname))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                print fname + ' written.'
                self.data_to_save = self.mid_res
                with open(fname, 'wb') as f:
                    pc.dump(self.data_to_save, f)
            if self.flag_callback_plot:
                if not os.path.exists(os.path.dirname(fname)):
                    try:
                        os.makedirs(os.path.dirname(fname))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                self.save_snapshot(xres_mid, fname)
            self.iter = self.iter + 1
            return [0]

    def __init__(self, *args, **kwargs):
        """
        @brief      Options for optimization
        
        @param      self    Optimization object
        @param      args    Optimization options
        @param      kwargs  Optimization options
        """
        data_out.__init__(self, *args, **kwargs)
        self.opt_opt = {'solver': 'ipopt',
                        'datafile_name': 'output_file',
                        'callback_steps': 100,
                        'method': 'not_thesis',
                        't_ind': 38,
                        't_int': 1,
                        'flag_depthweighted': True,
                        'sigma': 0.1,
                        'p_dyn': 10,
                        'flag_lift_mask': True,
                        'flag_data_mask': True,
                        'flag_write_output': True,
                        'flag_parallel': False,
                        'flag_callback': True,
                        'flag_callback_plot': False,
                        'solver': 'ipopt',
                        'hessian': 'exact',
                        'linsol': 'ma57'
                        }
        self.opt_opt.update(kwargs)
        self.solver = self.opt_opt['solver']
        self.datafile_name = self.opt_opt['datafile_name']
        self.callback_steps = self.opt_opt['callback_steps']
        self.t_ind = self.opt_opt['t_ind']
        self.t_int = self.opt_opt['t_int']
        self.sigma_value = self.opt_opt['sigma']
        self.method = self.opt_opt['method']
        self.flag_depthweighted = self.opt_opt['flag_depthweighted']
        self.flag_lift_mask = self.opt_opt['flag_lift_mask']
        self.flag_parallel = self.opt_opt['flag_parallel']
        self.flag_data_mask = self.opt_opt['flag_data_mask']
        self.flag_callback = self.opt_opt['flag_callback']
        self.flag_callback_plot = self.opt_opt['flag_callback_plot']
        self.flag_callback_output = self.opt_opt['flag_callback_output']
        self.flag_sparsity_pattern = self.opt_opt['flag_sparsity_pattern']
        self.flag_total = self.opt_opt['flag_total']
        self.flag_temporal = self.opt_opt['flag_temporal']
        self.flag_init = self.opt_opt['flag_init']
        self.flag_min_norm = self.opt_opt['flag_min_norm']
        self.flag_tv = self.opt_opt['flag_tv']
        self.flag_background = self.opt_opt['flag_background']
        self.p_solver = self.opt_opt['solver']
        self.p_hessian = self.opt_opt['hessian']
        self.p_linsol = self.opt_opt['linsol']
        self.p_dyn = self.opt_opt['p_dyn']
        # ######################## #
        #     Problem  setup       #
        # ######################## #
        print self.opt_opt
        self.y = self.data.electrode_rec[:, self.t_ind:self.t_ind + self.t_int]
        self.y_size = self.y.shape[0]
        self.t_size = self.y.shape[1]
        self.x_size = self.voxels[0, :].flatten().shape[0]
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        if self.flag_depthweighted:
            dw = self.cmp_weight_matrix(fwd)
            self.fwd = ca.MX(np.dot(fwd, dw))
        else:
            self.fwd = ca.MX(fwd)

    def initialize_variables(self):
        """
        @brief      Initialization for the optimization problem
        
        @param      self  The object
        
        @return     { None }
        """
        self.w0 = self.w(0)
        csd = self.get_ground_truth(method = 'modified')[0]
        if self.method == 'thesis':
            tmp_s0 = np.random.randn(self.s.shape[0], self.s.shape[1])
            for i in range(self.s.shape[0]):
                tmp_s0[i, :] = tmp_s0[i, :] / np.linalg.norm(tmp_s0[i, :])
            self.w0['s'] = tmp_s0
            print 'initialization: Thesis'
        print self.method
        if self.method == 'thesis' or self.method == 'mask':
            self.gt = np.reshape(csd[:,self.t_ind:self.t_ind+self.t_int], (self.w0['a'].shape))
            # tmp_m0 = np.random.rand(self.m.shape[0])
            tmp_m0 = np.ones(self.m.shape[0])*0.5 + np.random.randn(self.m.shape[0])/50
            print tmp_m0
            tmp_a0 = np.random.randn(self.a.shape[0], self.a.shape[1])/50
            print 'initialization: Thesis or Mask'
            if self.flag_init == 'rand':
                self.w0['m'] = tmp_m0
                self.w0['a'] = tmp_a0
            if self.flag_init == 'gt':
                self.w0['m'] = np.where(np.abs(self.gt) > 1e-3, 1, 0).T[0]
                self.w0['a'] = self.gt
            if self.flag_init == 'randm':
                self.w0['a'] = self.gt
                self.w0['m'] = tmp_m0
            if self.flag_init == 'mask':
                self.w0['a'] = tmp_a0
                self.w0['m'] = np.where(np.abs(self.gt) > 1e-3, 1, 0).T[0]
        if self.method == 'slack':
            self.gt = np.reshape(csd[:,self.t_ind:self.t_ind+self.t_int], (self.w0['x'].shape))
            if self.flag_init == 'rand':
                self.w0['x'] = np.random.randn(self.gt)
        if self.method == '2p':
            self.gt = np.reshape(csd[:,self.t_ind:self.t_ind+self.t_int], (self.w0['xs_pos'].shape))
            pos_charges = np.zeros(self.w0['xs_pos'].shape)
            neg_charges = np.zeros(self.w0['xs_neg'].shape)
            pos_charges[self.gt > 0] = self.gt[self.gt > 0]
            neg_charges[self.gt < 0] = self.gt[self.gt < 0]
            self.w0['xs_pos'] = pos_charges
            self.w0['xs_neg'] = neg_charges

    def minimize_function(self):
        """
        @brief      Main Minimization routine
        
        @param      self  Optimization Object
        
        @return     { None }
        """
        self.str_shape = self.w(0)
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)
        # Initialize 
        self.initialize_variables()
        # Create NLP
        self.nlp = {"x": self.w, "f": self.f, "g": self.g}
        # NLP solver options
        if self.p_solver == 'ipopt':
            self.opts = {"ipopt.max_iter": 100000,
                         "ipopt.hessian_approximation": self.p_hessian,
                         }
            self.opts["ipopt.fixed_variable_treatment"] = 'relax_bounds'
            self.opts["ipopt.tol"] = 1e-5
            if self.p_linsol[:2] == 'MA':
                print "###################"
                print "Using linear solver"
                print "###################"
                self.opts["ipopt.linear_solver"] = self.p_linsol
        elif self.p_solver == 'sqp':
            self.opts = {
                "qpsol": "qpoases"
            }
        if self.flag_callback:
            print self.w.shape[0]
            print self.g.shape[0]
            self.mycallback = opt_out.MyCallback('mycallback', self.w.shape[0], self.g.shape[0], 0,
                                                 opts={'filename': self.datafile_name,
                                                       'flag_callback_plot': self.flag_callback_plot,
                                                       'flag_callback_output': self.flag_callback_output,
                                                       'str_shape': self.str_shape,
                                                       'data_cb': self.data,
                                                       'p_method': self.method,
                                                       'gt': self.gt,
                                                       'voxels_cb': self.voxels})
            self.opts["iteration_callback"] = self.mycallback
            self.opts["iteration_callback_step"] = self.callback_steps
        # self.opts["ipopt.fixed_variable_treatment"] = 'make_constraint'
        # Create solver
        print "Initializing the solver"
        if self.p_solver == 'ipopt':
            self.solver = ca.nlpsol("solver", "ipopt", self.nlp, self.opts)
        elif self.p_solver == 'sqp':
            self.solver = ca.qpsol("solver", "qpoases", self.nlp)
        # Solve NLP
        self.args = {}
        self.args["x0"] = self.w0
        self.args["lbx"] = self.lbx
        self.args["ubx"] = self.ubx
        self.args["lbg"] = self.lbg
        self.args["ubg"] = self.ubg
        self.res = self.solver(**self.args)
        self.res_struct = self.w(self.res['x'])

    def cmp_dx(self, smooth_entity, i, j, k, t, h=1.):
        """
        @brief      Computes the forward difference along x dimension.
        
        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }
        
        @return     { returns the x derivative for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if i == 0:
            # dx = 0 (1st order)
            dx = 0
        elif i == 1:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind2 = np.ravel_multi_index((i + 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x0) x0 instead of x_2 (2nd order)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind2, t] - x[ind0, t])) / (12 * h)
        elif i == ni - 1:
            # dx = 0
            dx = 0
        elif i == ni - 2:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind0, t] - x[ind_2, t])) / (12 * h)
        else:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind2 = np.ravel_multi_index((i + 2, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind2, t] - x[ind_2, t])) / (12 * h)
        return dx

    def cmp_dy(self, smooth_entity, i, j, k, t, h=1.):
        """
        @brief      Computes the forward difference along y dimension.
        
        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }
        
        @return     { returns the y derivative for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if j == 0:
            # dx = 0 (1st order)
            dy = 0
        elif j == 1:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind2 = np.ravel_multi_index((i, j + 2, k), vx.shape)
            # dy = 8(x1-x_1)-1(x2-x0) x0 instead of x_2 (2nd order)
            dy = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind2, t] - x[ind0, t])) / (12 * h)
        elif j == nj - 1:
            # dy = 0
            dy = 0
        elif j == nj - 2:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
            # dy = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dy = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind0, t] - x[ind_2, t])) / (12 * h)
        else:
            if nj < 5:
                print "Warning: 1st order differences due to size"
                ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
                ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
                # dy = (x1-x_1)/2h
                dy = (8 * (x[ind1, t] - x[ind_1, t])) / (2 * h)
            else:
                ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
                ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
                ind2 = np.ravel_multi_index((i, j + 2, k), vx.shape)
                ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
                # dy = 8(x1-x_1)-1(x2-x_2)
                dy = (8 * (x[ind1, t] - x[ind_1, t]) -
                      1 * (x[ind2, t] - x[ind_2, t])) / (12 * h)
        return dy

    def cmp_dz(self, smooth_entity, i, j, k, t, h=1.):
        """
        @brief      Computes the forward difference along z dimension.
        
        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }
        
        @return     { returns the z derivative for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if k == 0:
            # dz = 0 (1st order)
            dz = 0
        elif k == 1:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind2 = np.ravel_multi_index((i, j, k + 2), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x0) x0 instead of x_2 (2nd order)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind2, t] - x[ind0, t])) / (12 * h)
        elif k == nk - 1:
            # dx = 0
            dz = 0
        elif k == nk - 2:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind0, t] - x[ind_2, t])) / (12 * h)
        else:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind2 = np.ravel_multi_index((i, j, k + 2), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1 * (x[ind2, t] - x[ind_2, t])) / (12 * h)
        return dz

    def cmp_gradient(self, smooth_entity, flag_tmp_smooth=False, h=1., flag_second=False):
        """
        @brief      Computes the forward difference for the whole volume.
        
        @param      self             {The optimization object}
        @param      smooth_entity    {The image to smooth}
        @param      flag_tmp_smooth  {The flag temporary smooth}
        @param      h                {discretization step}
        @param      flag_second      {The flag for second order approxiamion}
        @param      t     {time index}
        
        @return     { returns the gradient for a the whole image }
        """
        # initials
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        nv = ni * nj * nk
        if x.shape[0] != nv:
            nt = nv / x.shape[0]
        else:
            nt = 1
        # print "Time point(s): ", nt
        # loop over the voxels
        for t in range(nt):
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        ind = np.ravel_multi_index((i, j, k), vx.shape)
                        if ind == 0:
                            grad_mtr = [sum([self.cmp_dx(smooth_entity, i, j, k, t, h)]) + 1e-20,
                                        sum([self.cmp_dy(smooth_entity,
                                                         i, j, k, t, h)]) + 1e-20,
                                        sum([self.cmp_dz(smooth_entity, i, j, k, t, h)]) + 1e-20]
                        else:
                            grad_mtr = ca.horzcat(grad_mtr,
                                                  ca.vertcat(sum([self.cmp_dx(smooth_entity, i, j, k, t, h)]),
                                                             sum([self.cmp_dy(
                                                                 smooth_entity, i, j, k, t, h)]),
                                                             sum([self.cmp_dz(smooth_entity, i, j, k, t, h)])) + 1e-20)
        if flag_tmp_smooth:
            # compute temporal gradient
            print "Temporal smoothness enforced."
        return grad_mtr

    def cmp_fwd_dx(self, smooth_entity, flag_average, i, j, k, t, h=1.):
        """
        @brief      Computes the forward difference along x dimension.
        
        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      flag_average   The flag to get mask average
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }
        
        @return     { returns the gradient for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if i == ni - 1:
            ind1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        else:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if flag_average:
            dx = (x[ind1, t] + x[ind0, t]) / 2.
        else:
            dx = (x[ind1, t] - x[ind0, t]) / h
        return dx

    def cmp_fwd_dy(self, smooth_entity, flag_average, i, j, k, t, h=1.):
        """
        Computes the forward difference along y dimension.

        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      flag_average   The flag to get mask average
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }

        @return     { returns the gradient for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if j == nj - 1:
            ind1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        else:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if flag_average:
            dy = (x[ind1, t] + x[ind0, t]) / 2.
        else:
            dy = (x[ind1, t] - x[ind0, t]) / h
        return dy

    def cmp_fwd_dz(self, smooth_entity, flag_average, i, j, k, t, h=1.):
        """
        @brief      Computes the forward difference along z dimension.
        
        @param      self           The optimization object
        @param      smooth_entity  The image to smooth
        @param      flag_average   The flag to get mask average
        @param      i              { pixel index for x - dimension}
        @param      j              { pixel index for y - dimension}
        @param      k              { pixel index for z - dimension}
        @param      t              { time index }
        @param      h              { discretization step }
        
        @return     { returns the gradient for a single pixel }
        """
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if k == nk - 1:
            ind1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        else:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind0 = np.ravel_multi_index((i, j, k), vx.shape)
        if flag_average:
            dz = (x[ind1, t] + x[ind0, t]) / 2.
        else:
            dz = (x[ind1, t] - x[ind0, t]) / h
        return dz

    def cmp_fwd_diff(self, smooth_entity, flag_average=False, flag_tmp_smooth=False, h=1., flag_second=False):
        """
        @brief      { Compute forward difference }
        
        @param      self             The object
        @param      smooth_entity    The smooth entity
        @param      flag_average     The flag average
        @param      flag_tmp_smooth  The flag temporary smooth
        @param      h                { parameter_description }
        @param      flag_second      The flag second
        
        @return     { None }
        """
        # initials
        x = smooth_entity
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        nv = ni * nj * nk
        if flag_average:
            epsilon = 0
        else:
            epsilon = 1e-20
        if x.shape[0] != nv:
            nt = nv / x.shape[0]
        else:
            nt = 1
        # loop over the voxels
        for t in range(nt):
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        if i == j == k == 0:
                            grad_fwd = ca.vertcat(sum([self.cmp_fwd_dx(smooth_entity, flag_average, i, j, k, t, h)]) + epsilon,
                                                  sum([self.cmp_fwd_dy(
                                                      smooth_entity, flag_average, i, j, k, t, h)]) + epsilon,
                                                  sum([self.cmp_fwd_dz(smooth_entity, flag_average, i, j, k, t, h)]) + epsilon)
                        else:
                            grad_fwd = ca.horzcat(grad_fwd,
                                                  ca.vertcat(sum([self.cmp_fwd_dx(smooth_entity, flag_average, i, j, k, t, h)]),
                                                             sum([self.cmp_fwd_dy(
                                                                 smooth_entity, flag_average, i, j, k, t, h)]),
                                                             sum([self.cmp_fwd_dz(smooth_entity, flag_average, i, j, k, t, h)])) + epsilon)
        if flag_tmp_smooth:
            # compute temporal gradient
            print "Temporal smoothness enforced."
        if self.flag_parallel:
            print smooth_entity.T.shape, grad_fwd.shape
            # gr = ca.Function('gr',[smooth_entity.T],[grad_fwd])
            # return gr_map(smooth_entity)
            return gr
        else:
            return grad_fwd

    def write_casadi_structure(self, struct_to_save):
        """
        @brief      { write casadi structure }

        @param      self            The object.
        @param      struct_to_save  The struct to save

        @return     { None }
        """
        fname = '../results/' + self.datafile_name + '/' + 'struct.cs'
        struct_to_save.save(fname)

    def load_casadi_structure(self, fname):
        """
        @brief      { function_description }

        @param      self   The object.
        @param      fname  The fname

        @return     { None }
        """
        return ca.tools.struct_load(fname)

    def optimize_waveform(self, x):
        """
        @brief      fit waveform to a bimodal alpha function
        
        @param      self  The object
        @param      x     { parameter_description }
        
        @return     { None }
        """
        srate = self.data.srate
        fit_data = self.data.cell_csd[i, 38:]
        # fit_data = x
        tlin = np.linspace(
            1. / srate, (fit_data.shape[0] - 1) / srate, fit_data.shape[0] - 1. / srate)
        t = ca.MX.sym("t")
        t1 = ca.MX.sym("t1")
        t2 = ca.MX.sym("t2")
        a = ca.MX.sym("a")
        r = ca.vertcat(t1, t2, a)
        f = (ca.exp(-t * t1) * t * t1 * t1 - ca.exp(-t * t2) * t * t2 * t2) * a
        F = ca.Function("F", [r, t], [f])
        Y = [(fit_data[i] - F(r, tlin[i]))**2 for i in range(fit_data.shape[0])]
        nlp_root = {"x": r, "f": sum(Y)}
        root_solver = ca.nlpsol("solver", "ipopt", nlp_root,
                                {'ipopt.file_print_level': 0,
                                 'ipopt.print_level': 0,
                                 'ipopt.print_timing_statistics': 'no'})
        r0 = [1.e3, 2.e3, 0.]
        args = {}
        args["x0"] = r0
        args["lbx"] = ca.vertcat([-ca.inf, -ca.inf, -ca.inf])
        args["ubx"] = ca.vertcat([ca.inf, ca.inf, ca.inf])
        res = root_solver(**args)
        return [F(res["x"], tlin[i]) for i in range(fit_data.shape[0])]

    def set_optimization_variables_slack(self):
        """
        @brief      Variables for the lifted version
        
        @param      self  The object
        
        @return     { None }
        """
        self.w = struct_symMX([entry("x", shape=(self.x_size, self.t_int)),
                               entry("xs", shape=(self.x_size)),
                               entry("ys", shape=(self.y.shape))])
        self.x, self.xs, self.ys = self.w[...]
        self.g = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)

    def add_data_costs_constraints_slack(self):
        """
        @brief      Computes objective function f With lifting variable ys
                    constraints
        
        @param      self  The object
        
        @return     { None }
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] - ca.dot(self.fwd[i, :].T, self.x[:, ti]))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_l1_costs_constraints_slack(self):
        """
        @brief      add slack l1 constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        for j in range(self.w['xs'].shape[0]):
            tmp = 0
            for tj in range(self.t_int):
                tmp += self.w['x'][j, tj]**2
            self.f += self.sigma_value * (self.xs[j])
            self.g.append(-self.xs[j] - tmp)
            self.g.append(-self.xs[j] + tmp)
            self.g.append(-self.xs[j])
            self.lbg.append(-ca.inf)
            self.lbg.append(-ca.inf)
            self.lbg.append(-ca.inf)
            self.ubg.append(0.)
            self.ubg.append(0.)
            self.ubg.append(0.)

    def solve_ipopt_multi_measurement_slack(self):
        """
        @brief      MMV L1
        
        @param      self  The object
        
        @return     { None }
        """
        t0 = time.time()
        self.set_optimization_variables_slack()
        self.add_data_costs_constraints_slack()
        self.add_l1_costs_constraints_slack()
        t1 = time.time()
        print "Set constraints in %.3f seconds" % (t1 - t0)
        t0 = time.time()
        self.minimize_function()
        t1 = time.time()
        print "Minimize in %.3f seconds" % (t1 - t0)
        if self.opt_opt['flag_write_output']:
            save_this = {}
            save_this['res'] = self.res
            save_this['opts'] = self.opt_opt
            self.write_with_pickle(save_this)
            self.write_casadi_structure(self.res_struct)
        self.xres = self.res_struct['x'].full()

    def set_optimization_variables_2p(self):
        """
        @brief      x is divided into negative and positive elements this
                    function overwrites the initialized optimization variables
        
        @param      self  The object
        
        @return     { None }
        """
        self.w = struct_symMX([entry("xs_pos", shape=(self.x_size, self.t_int)),
                               entry("xs_neg", shape=(self.x_size, self.t_int)),
                               entry("ys", shape=(self.y.shape))])
        self.xs_pos, self.xs_neg, self.ys = self.w[...]
        self.g = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)

    def add_data_costs_constraints_2p(self):
        """
        @brief      Computes objective function f With lifting variable ys
                    constraints
        
        @param      self  The object
        
        @return     { None }
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] - ca.dot(self.fwd[i, :].T, (self.xs_pos[:, ti] - self.xs_neg[:, ti])))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_l1_costs_constraints_2p(self):
        """
        @brief      add slack l1 constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        for j in range(self.xs_pos.shape[0]):
            tmp_pos = 0
            tmp_neg = 0
            for tj in range(self.t_int):
                tmp_pos += self.xs_pos[j, tj]
                tmp_neg += self.xs_neg[j, tj]
                self.g.append(-self.xs_pos[j, tj])
                self.g.append(-self.xs_neg[j, tj])
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.ubg.append(0.)
                self.ubg.append(0.)
            self.f += self.sigma_value * (tmp_pos + tmp_neg)

    def solve_ipopt_multi_measurement_2p(self):
        """
        @brief      Reform source space x as the difference of x+ - x-
        
        @param      self  The object
        
        @return     { None }
        """
        t0 = time.time()
        self.set_optimization_variables_2p()
        self.add_data_costs_constraints_2p()
        self.add_l1_costs_constraints_2p()
        t1 = time.time()
        print "Set constraints in %.3f seconds" % (t1 - t0)
        t0 = time.time()
        self.minimize_function()
        t1 = time.time()
        print "Minimize in %.3f seconds" % (t1 - t0)
        if self.opt_opt['flag_write_output']:
            save_this = {}
            save_this['res'] = self.res
            save_this['opts'] = self.opt_opt
            self.write_with_pickle(save_this)
            self.write_casadi_structure(self.res_struct)
        self.xres = self.res_struct['xs_pos'].full() - self.res_struct['xs_neg'].full()

    def set_optimization_variables_thesis(self):
        """
        @brief      thesis implementation
        
        @param      self  The object
        
        @return     { None }
        """
        self.w = struct_symMX([entry("a", shape=(self.x_size, self.t_int)),
                               entry("m", shape=(self.x_size)),
                               entry("s", shape=(self.x_size, 3)),
                               entry("ys", shape=(self.y.shape)),
                               ])
        self.a, self.m, self.s, self.ys = self.w[...]
        self.g = []
        self.gBlock = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.sigma = ca.MX.sym('sigma')
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)
        # self.lbx['m'] = 0
        # self.ubx['m'] = 1

    def add_data_costs_constraints_thesis(self):
        """
        @brief      Computes objective function f With lifting variable ys
                    constraints
        
        @param      self  The object
        
        @return     { None }
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                # self.f += (self.y[i, ti] - self.ys[i, ti])**2
                if self.flag_data_mask:
                    # self.g.append(self.ys[i, ti] - ca.dot(self.fwd[i, :].T, self.a[:, ti] * self.m))
                    self.g.append(self.y[i, ti] - ca.dot(self.fwd[i, :].T, self.a[:, ti] * self.m))
                else:
                    # self.g.append(self.ys[i, ti] - ca.dot(self.fwd[i, :].T, self.a[:, ti]))
                    self.g.append(self.y[i, ti] - ca.dot(self.fwd[i, :].T, self.a[:, ti]))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_ordered_constraints_mask(self):
        """
        @brief      Adds an ordered constraints mask.
        
        @param      self  The object
        
        @return     { None }
        """
        

    def add_l1_costs_constraints_thesis(self):
        """
        @brief      add slack l1 constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        print self.sigma_value*2
        for j in range(self.m.shape[0]):
            self.f += self.sigma_value * (self.m[j])
            if self.flag_lift_mask:
                self.g.append(1 - (self.m[j]**2 + (1 - self.m[j])**2)**0.5)
                self.lbg.append(0)
                self.ubg.append(0)

    def add_background_costs_constraints_thesis(self):
        """
        @brief      add background constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        for b in range(self.m.shape[0]):
            tmp = 0
            for tb in range(self.t_int):
                tmp += self.a[b, tb]**2
            self.g.append(tmp * (1 - self.m[b]))
            self.lbg.append(0)
            self.ubg.append(0)

    def add_tv_mask_costs_constraints_thesis(self):
        """
        @brief      add smoothness constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        if self.flag_tv == 'cent':
            grad_m = self.cmp_gradient(self.m, False)
        if self.flag_tv == 'fwd':
            grad_m = self.cmp_fwd_diff(self.m, False)
        for cm in range(self.m.shape[0]):
            self.g.append(ca.dot(grad_m[:, cm], grad_m[:, cm]))
            #self.g.append(ca.dot(grad_m[:, cm], grad_m[:, cm]))
            self.lbg.append(0)
            # self.ubg.append(3**0.5)
            self.ubg.append(3)

    def add_smoothness_costs_constraints_thesis(self):
        """
        @brief      Adds a smoothness costs constraints thesis.
        
        @param      self  The object
        
        @return     { None }
        """
        average_mask = self.cmp_fwd_diff(self.m, True)
        average_sx = self.cmp_fwd_diff(self.s[:, 0], True)[0, :]
        average_sy = self.cmp_fwd_diff(self.s[:, 1], True)[1, :]
        average_sz = self.cmp_fwd_diff(self.s[:, 2], True)[2, :]
        for tb in range(self.t_int):
            tmp = self.cmp_fwd_diff(self.a[:, tb], False)
            for b in range(self.x_size):
                average_s = ca.vertcat(average_mask[0, b] * average_sx[b],
                                       average_mask[1, b] * average_sy[b],
                                       average_mask[2, b] * average_sz[b])
                self.g.append(ca.dot(average_s, tmp[:, b])**2)
                self.lbg.append(0)
                self.ubg.append(0)

    def add_s_magnitude_costs_constraints_thesis(self):
        """
        @brief      add smoothness constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        for b in range(self.x_size):
            self.g.append(ca.dot(self.s[b, :], self.s[b, :]))
            self.lbg.append(1)
            self.ubg.append(1)

    def add_temporal_smoothness_constraints(self):
        """
        @brief      Adds a temporal smoothness constraints.
        
        @param      self  The optimization object
        
        @return     { None }
        """
        for ts in range(self.t_size):
            for tx in range(self.x_size):
                if ts != self.t_size-1:
                    self.g.append((self.a[tx, ts+1]-self.a[tx, ts])**2)
                    self.lbg.append(0)
                    self.ubg.append(20)

    def add_total_charge_constraints(self):
        """
        @brief      Adds a total charge constraints.
        
        @param      self  The optimization object
        
        @return     { None }
        """
        for ts in range(self.t_size):
            self.g.append(sum([self.a[:, ts]]))
            self.lbg.append(0)
            self.ubg.append(20)

    def add_min_norm(self):
        """
        @brief      Adds a minimum normalize.
        
        @param      self  The object
        
        @return     { description_of_the_return_value }
        """
        for ts in range(self.t_size):
            self.f += ca.dot(self.a[:, ts], self.a[:, ts])
            
    def add_s_smooth_costs_constraints_thesis(self):
        """
        @brief      add smoothness constraints with lifting variables
        
        @param      self  The object
        
        @return     { None }
        """
        grad = []
        grad_x = self.cmp_fwd_diff(self.s[:, 0], False)
        grad_y = self.cmp_fwd_diff(self.s[:, 1], False)
        grad_z = self.cmp_fwd_diff(self.s[:, 2], False)
        average_mask = self.cmp_fwd_diff(self.m, True)
        for b in range(self.s.shape[0]):
            self.g.append(average_mask[0, b] * ca.dot(grad_x[:, b], grad_x[:, b]) +
                          average_mask[1, b] * ca.dot(grad_y[:, b], grad_y[:, b]) +
                          average_mask[2, b] * ca.dot(grad_z[:, b], grad_z[:, b]))
            self.lbg.append(0)
            self.ubg.append(self.p_dyn)

    def solve_ipopt_multi_measurement_thesis(self):
        """
        @brief      Reform source space x as the difference of x+ - x-
        
        @param      self  The object
        
        @return     { None }
        """
        self.set_optimization_variables_thesis()
        t0 = time.time()
        self.add_data_costs_constraints_thesis()
        self.add_l1_costs_constraints_thesis()
        self.add_background_costs_constraints_thesis()
        self.add_tv_mask_costs_constraints_thesis()
        self.add_smoothness_costs_constraints_thesis()
        self.add_s_magnitude_costs_constraints_thesis()
        self.add_s_smooth_costs_constraints_thesis()
        t1 = time.time()
        print "Set constraints in %.3f seconds" % (t1 - t0)
        t0 = time.time()
        self.minimize_function()
        t1 = time.time()
        print "Minimize in %.3f seconds" % (t1 - t0)
        if self.opt_opt['flag_write_output']:
            save_this = {}
            save_this['res'] = self.res
            save_this['opts'] = self.opt_opt
            self.write_with_pickle(save_this)
            self.write_casadi_structure(self.res_struct)
        self.xres = self.res_struct['a'].full()
        self.sres = self.res_struct['s'].full()

    def set_optimization_variables_only_mask(self):
        """
        @brief      thesis implementation
        
        @param      self  The optimization object
        
        @return     { None }
        """
        self.w = struct_symMX([entry("a", shape=(self.x_size, self.t_int)),
                               entry("m", shape=(self.x_size)),
                               # entry("ys", shape=(self.y.shape)),
                               ])
        # self.a, self.m, self.ys = self.w[...]
        self.a, self.m = self.w[...]
        self.g = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.sigma = ca.MX.sym('sigma')
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)
        self.lbx['m'] = 0
        self.ubx['m'] = 1
        if self.flag_sparsity_pattern:
            self.w = struct_symMX([(entry("a", repeat=[self.x_size, self.t_int]),
                       entry("m", repeat=self.x_size)),
                       entry("ys", shape=self.y.shape)
                       ])
            self.a, self.m, self.ys = self.w[...]
            self.g = []
            self.lbg = []
            self.ubg = []
            self.f = 0
            self.sigma = ca.MX.sym('sigma')
            self.lbx = self.w(-ca.inf)
            self.ubx = self.w(ca.inf)
            self.lbx['m'] = 0
            self.ubx['m'] = 1

    def solve_ipopt_multi_measurement_only_mask(self):
        """
        @brief      Reform source space x as the difference of x+ - x-
        
        @param      self  The optimization object
        
        @return     { None }
        """
        self.set_optimization_variables_only_mask()
        t0 = time.time()
        if self.flag_background:
            self.add_background_costs_constraints_thesis()
        self.add_data_costs_constraints_thesis()
        self.add_l1_costs_constraints_thesis()
        if self.flag_temporal:
            self.add_temporal_smoothness_constraints()
        if self.flag_total:
            self.add_total_charge_constraints()
        if self.flag_tv != 'none':
            self.add_tv_mask_costs_constraints_thesis()
        t1 = time.time()
        if self.flag_min_norm:
            self.add_min_norm()
        print "Set constraints in %.3f seconds" % (t1 - t0)
        t0 = time.time()
        self.minimize_function()
        t1 = time.time()
        print "Minimize in %.3f seconds" % (t1 - t0)
        if self.opt_opt['flag_write_output']:
            save_this = {}
            save_this['res'] = self.res
            save_this['opts'] = self.opt_opt
            self.write_with_pickle(save_this)
            self.write_casadi_structure(self.res_struct)
        self.xres = self.res_struct['a'].full() * self.res_struct['m'].full() 

    def get_ground_truth(self, method = 'shephard'):
        """
        @brief      Get the ground truth.
        
        @param      self    The optimization object
        @param      method  The method
        
        @return     Ground truth.
        """
        data = self.data
        # mask reconstruction volume
        vx, vy, vz = self.voxels
        rx, ry, rz = vx, vy, vz
        vx, vy, vz = vx.flatten(), vy.flatten(), vz.flatten()
        voxel_width = self.options['p_vres'] / 2.
        vx_min, vy_min, vz_min = vx - voxel_width, vy - voxel_width, vz - voxel_width
        vx_max, vy_max, vz_max = vx + voxel_width, vy + voxel_width, vz + voxel_width
        # result
        xmin, xmax = np.min(vx), np.max(vx)
        ymin, ymax = np.min(vy), np.max(vy)
        zmin, zmax = np.min(vz), np.max(vz)
        ind_cell = ((xmin <= data.cell_pos[:, 0]) & (xmax >= data.cell_pos[:, 0]) &
                    (ymin <= data.cell_pos[:, 1]) & (ymax >= data.cell_pos[:, 1]) &
                    (zmin <= data.cell_pos[:, 2]) & (zmax >= data.cell_pos[:, 2]))
        vis_cell_pos = data.cell_pos[ind_cell.nonzero()[0], :]
        vis_cell_csd = data.cell_csd[ind_cell.nonzero()[0], :]
        ind_in_vox = []
        for iv in range(vx_min.shape[0]):
            ind_in_vox.append((vx_min[iv] <= vis_cell_pos[:, 0]) & (vis_cell_pos[:, 0] <= vx_max[iv]) &
                              (vy_min[iv] <= vis_cell_pos[:, 1]) & (vy_max[iv] >= vis_cell_pos[:, 1]) &
                              (vz_min[iv] <= vis_cell_pos[:, 2]) & (vz_max[iv] >= vis_cell_pos[:, 2]))
        ind_in_vox = np.array(ind_in_vox)
        nnz_ind_in_vox = np.nonzero(np.sum(ind_in_vox, 1))[0]
        vox_csd = np.zeros((vx.shape[0], vis_cell_csd.shape[1]))
        vox_pt = np.zeros(vx.shape[0])
        if method == 'shephard':
            vox_cell_pos = vis_cell_pos
            vox_cell_csd = vis_cell_csd
            vox_pos = np.array([vx, vy, vz]).T
            vox_dis = 1./cdist(vox_cell_pos, vox_pos)**2
            vox_csd = np.dot(vox_dis.T, vox_cell_csd) / np.sum(vox_dis.T, axis = 1).reshape(vox_dis.T.shape[0], 1)
        elif method == 'modified':
            for ivv in nnz_ind_in_vox:
                vox_cell_pos = vis_cell_pos[ind_in_vox[ivv, :]]
                vox_cell_csd = vis_cell_csd[ind_in_vox[ivv, :]]
                vox_pos = [vx[ivv], vy[ivv], vz[ivv]]
                vox_dis = 1. / \
                    np.sum(np.abs(vox_cell_pos - vox_pos)**2, axis=-1)
                vox_csd[ivv,:] = np.dot(vox_cell_csd[:, :],
                                      vox_dis) / np.sum(vox_dis)
        return [vox_csd]

    def get_kcsd_estimate(self):
        """
        @brief      Gets the kcsd estimate.

        @param      self  The optimization object

        @return     The kcsd estimate.
        """
        from pykCSD.pykCSD import KCSD
        elec_pos = (self.electrode_pos[:, :])
        pots = np.array(self.data.electrode_rec[
                        :, 38].flatten().reshape(64, 1))
        params = {
            'gdX': 5,
            'gdY': 5,
            'gdZ': 5,
            'n_sources': 64,
        }
        k = KCSD(elec_pos, pots, params)
        k.estimate_pots()
        k.estimate_csd()
        k.plot_all()
