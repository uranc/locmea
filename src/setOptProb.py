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

    def solve_ipopt_slack_cost(self):
        # ######################## #
        #    Problem parameters    #
        # ######################## #
        self.t_ind = 30
        self.t_int = 1
        # ######################## #
        #   Optimization problems  #
        # ######################## #
        self.method = 'grad'
        # ######################## #
        #  Optimization variables  #
        # ######################## #
        self.y = ca.SX(self.data.electrode_rec[
                       :, self.t_ind:self.t_ind+self.t_int])
        self.x = ca.SX.sym('x', self.voxels[0, :].flatten().shape[0])
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        dw = self.cmp_weight_matrix(fwd)
        dfwd = np.dot(fwd, dw)
        self.fwd = ca.SX(dfwd)
        # self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.sim = ca.Function('sim', [self.x], [ca.mtimes(self.fwd, self.x)])
        self.ls = ca.Function('ls', [self.x], [self.y-self.sim([self.x])[0]])
        self.f = 0
        self.g = ca.norm_1(self.x)
        # Constraint
        self.f = ca.dot(self.ls([self.x])[0], self.ls([self.x])[0])
        # Bounds
        self.lbg = np.ones(self.g.shape[0])*0.
        self.ubg = np.ones(self.g.shape[0])*30.
        self.lbx = np.ones(self.x.shape[0])*-100.
        self.ubx = np.ones(self.x.shape[0])*100.
        # Initialize
        self.x0 = np.random.randn(self.x.shape[0])*0.
        # Create NLP
        self.nlp = {'x': self.x, 'f': self.f, 'g': self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, self.opts)
        # Solve NLP
        args = {}
        args["x0"] = self.x0
        args["lbx"] = ca.vertcat([self.lbx])
        args["ubx"] = ca.vertcat([self.ubx])
        args["lbg"] = ca.vertcat([self.lbg])
        args["ubg"] = ca.vertcat([self.ubg])
        self.res = self.solver(args)
        self.xres = self.res['x'].full().reshape(self.voxels[0, :, :, :].shape)

    def cmp_dx(self, i, j, k, t, h=1.):
        """
        cmp_dx
        """
        x = self.x
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
                  1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif i == ni - 1:
            # dx = 0
            dx = 0
        elif i == ni - 2:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind2 = np.ravel_multi_index((i + 2, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dx

    def cmp_dy(self, i, j, k, t, h=1.):
        """
        cmp_dy
        """
        x = self.x
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
            dy = (8*(x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif j == nj - 1:
            # dy = 0
            dy = 0
        elif j == nj - 2:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
            # dy = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dy = (8*(x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            if nj < 5:
                print "Warning: 1st order differences due to size"
                ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
                ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
                # dy = (x1-x_1)/2h
                dy = (8*(x[ind1, t] - x[ind_1, t]))/(2*h)
            else:
                ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
                ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
                ind2 = np.ravel_multi_index((i, j + 2, k), vx.shape)
                ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
                # dy = 8(x1-x_1)-1(x2-x_2)
                dy = (8*(x[ind1, t] - x[ind_1, t]) -
                      1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dy

    def cmp_dz(self, i, j, k, t, h=1.):
        """
        cmp_dz
        """
        x = self.x
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
                  1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif k == nk - 1:
            # dx = 0
            dz = 0
        elif k == nk - 2:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind2 = np.ravel_multi_index((i, j, k + 2), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) -
                  1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dz

    def cmp_gradient(self, flag_tmp_smooth=False, h=1., flag_second=True):
        """
        <F7>cmp_gradient
        """
        # initials
        x = self.x
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        nv = ni * nj * nk
        if x.shape[0] != nv:
            nt = nv / x.shape[0]
        else:
            nt = 1
        print "Time point(s): ", nt
        # loop over the voxels
        for t in range(nt):
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        ind = np.ravel_multi_index((i, j, k), vx.shape)
                        if ind == 0:
                            grad_mtr = ca.sumRows(
                                ca.sumRows(self.cmp_dx(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dy(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dz(i, j, k, t, h))**2)
                        else:
                            grad_mtr = ca.vertcat([grad_mtr, ca.sumRows(
                                ca.sumRows(self.cmp_dx(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dy(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dz(i, j, k, t, h))**2)])
        if flag_tmp_smooth:
            # compute temporal gradient
            print "Temporal smoothness enforced."
        return grad_mtr

    def optimize_waveform(self):
        """
        fit waveform to a biphasic alpha function
        """
        srate = self.data.srate
        fit_data = self.data.cell_csd[0, 30:]
        tlin = ca.SX(np.linspace(
                     0, (fit_data.shape[0]-1)/srate, fit_data.shape[0]))
        t = ca.SX.sym('t')
        t1 = ca.SX.sym('t1')
        t2 = ca.SX.sym('t2')
        a = ca.SX.sym('a')
        r = ca.vertcat([t1, t2, a])
        f = (ca.exp(-t*t1)*t*t1*t1 - ca.exp(-t*t2)*t*t2*t2)*a
        F = ca.Function('F', [r, t], [f])
        Y = [(fit_data[i]-F([r, tlin[i]])[0])**2
             for i in range(fit_data.shape[0])]
        nlp_root = {'x': r, 'f': sum(Y)**(1./2)}
        root_solver = ca.nlpsol("solver", "ipopt", nlp_root)
        r0 = [1.e3, 2.e3, 1.]
        args = {}
        args["x0"] = r0
        args["lbx"] = ca.vertcat([-ca.inf, -ca.inf, -ca.inf])
        args["ubx"] = ca.vertcat([ca.inf, ca.inf, ca.inf])
        res = root_solver(args)
        return [F([res['x'], tlin[i]])[0] for i in range(fit_data.shape[0])]

    def solve_ipopt_reformulate(self):
        # ######################## #
        #    Problem parameters    #
        # ######################## #
        self.t_ind = 30
        self.t_int = 1
        # ######################## #
        #   Optimization problems  #
        # ######################## #
        self.method = 'grad'
        # ######################## #
        #  Optimization variables  #
        # ######################## #
        self.ys = ca.MX.sym('ys', self.data.electrode_rec[
                            :, self.t_ind:self.t_ind+self.t_int].shape)
        self.y = ca.MX(self.data.electrode_rec[
                       :, self.t_ind:self.t_ind+self.t_int])
        self.x = ca.MX.sym('x', self.voxels[0, :].flatten().shape[0])
        self.xs = ca.MX.sym('xs', self.voxels[0, :].flatten().shape[0])
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        dw = self.cmp_weight_matrix(fwd)
        dfwd = np.dot(fwd, dw)
        self.fwd = ca.MX(dfwd)
        # self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.sim = ca.Function('sim', [self.x], [ca.mtimes(self.fwd, self.x)])
        self.ls = ca.Function('ls', [self.xs, self.ys],
                              [self.ys-self.sim([self.xs])[0]])
        self.f = 0
        self.g = []
        for i in range(self.y.shape[0]):
            self.f += (self.y[i] - self.ys[i])**2
            self.g.append(self.y[i] - ca.dot(self.fwd[i, :].T, self.x))
        self.g = ca.vertcat(self.g)
        # Bounds
        self.lbg = np.ones(self.y.shape[0])*0.
        self.ubg = np.ones(self.y.shape[0])*0.
        self.w = ca.vertcat([self.x, self.ys])
        self.lbx = np.ones(self.w.shape[0])*-ca.inf
        self.ubx = np.ones(self.w.shape[0])*ca.inf
        # Initialize
        self.w0 = np.random.randn(self.w.shape[0])*0.
        # Create NLP
        self.nlp = {'x': self.w, 'f': self.f, 'g': self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp)
        # Solve NLP
        args = {}
        args["x0"] = self.w0
        args["lbx"] = ca.vertcat([self.lbx])
        args["ubx"] = ca.vertcat([self.ubx])
        args["lbg"] = ca.vertcat([self.lbg])
        args["ubg"] = ca.vertcat([self.ubg])
        self.res = self.solver(args)
        self.xres = self.res['x'].full().reshape(self.voxels[0, :, :, :].shape)
