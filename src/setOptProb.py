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
                            grad_mtr = ca.vertcat(grad_mtr, ca.sumRows(
                                ca.sumRows(self.cmp_dx(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dy(i, j, k, t, h))**2 +
                                ca.sumRows(self.cmp_dz(i, j, k, t, h))**2))
        if flag_tmp_smooth:
            # compute temporal gradient
            print "Temporal smoothness enforced."
        return grad_mtr

    def optimize_waveform(self):
        """
        fit waveform to a bimodal alpha function
        """
        srate = self.data.srate
        fit_data = self.data.cell_csd[0, 30:]
        tlin = ca.MX(np.linspace(
                     0, (fit_data.shape[0]-1)/srate, fit_data.shape[0]))
        t = ca.MX.sym("t")
        t1 = ca.MX.sym("t1")
        t2 = ca.MX.sym("t2")
        a = ca.MX.sym("a")
        r = ca.vertcat([t1, t2, a])
        f = (ca.exp(-t*t1)*t*t1*t1 - ca.exp(-t*t2)*t*t2*t2)*a
        F = ca.Function("F", [r, t], [f])
        Y = [(fit_data[i]-F([r, tlin[i]])[0])**2
             for i in range(fit_data.shape[0])]
        nlp_root = {"x": r, "f": sum(Y)**(1./2)}
        root_solver = ca.nlpsol("solver", "ipopt", nlp_root)
        r0 = [1.e3, 2.e3, 1.]
        args = {}
        args["x0"] = r0
        args["lbx"] = ca.vertcat([-ca.inf, -ca.inf, -ca.inf])
        args["ubx"] = ca.vertcat([ca.inf, ca.inf, ca.inf])
        res = root_solver(**args)
        return [F([res["x"], tlin[i]])[0] for i in range(fit_data.shape[0])]

    def solve_ipopt_single_measurement(self):
        # ######################## #
        #    Problem parameters    #
        # ######################## #
        self.t_ind = 30
        self.t_int = 1
        # ######################## #
        #   Optimization problems  #
        # ######################## #
        self.method = "grad"
        # ######################## #
        #  Optimization variables  #
        # ######################## #
        self.ys = ca.MX.sym("ys", self.data.electrode_rec[
                            :, self.t_ind:self.t_ind+self.t_int].shape)
        self.y = self.data.electrode_rec[:, self.t_ind:self.t_ind+self.t_int]
        self.x = ca.MX.sym("x", self.voxels[0, :].flatten().shape[0])
        self.xs = ca.MX.sym("xs", self.voxels[0, :].flatten().shape[0])
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        dw = self.cmp_weight_matrix(fwd)
        self.fwd = np.dot(fwd, dw)
        self.sigma = 0
        # self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.f = 0
        self.w = []
        self.g = []
        # bounds
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        for i in range(self.y.shape[0]):
            self.f += (self.y[i] - self.ys[i])**2
            self.g.append(self.ys[i] - ca.dot(self.fwd[i, :], self.x))
            self.lbg.append(0)
            self.ubg.append(0)
            # self.w.append(self.ys[i])
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
        for j in range(self.x.shape[0]):
            self.f += self.sigma*(self.xs[j])
            self.g.append(-self.xs[j]-self.x[j])
            self.g.append(-self.xs[j]+self.x[j])
            self.g.append(-self.xs[j])
            self.lbg.append(-ca.inf)
            self.lbg.append(-ca.inf)
            self.lbg.append(-ca.inf)
            self.ubg.append(0)
            self.ubg.append(0)
            self.ubg.append(0)
            self.lbx.append(-ca.inf)
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
            self.ubx.append(ca.inf)
        # sigma bound
        self.w = ca.vertcat(self.ys, self.xs, self.x)
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)
        self.lbx = ca.vertcat(*self.lbx)
        self.ubx = ca.vertcat(*self.ubx)
        # Initialize
        self.w0 = ca.vertcat(np.zeros(self.w.shape[0]))
        # Create NLP
        self.nlp = {"x": self.w, "f": self.f, "g": self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000,
                     # "ipopt.linear_solver": 'pardisos',
                     "ipopt.hessian_approximation": "limited-memory"}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp)
        # Solve NLP
        self.args = {}
        self.args["x0"] = self.w0
        self.args["lbx"] = self.lbx
        self.args["ubx"] = self.ubx
        self.args["lbg"] = self.lbg
        self.args["ubg"] = self.ubg
        self.res = self.solver(self.args)
        self.xres = self.res["x"].full()[self.ys.shape[0]+self.xs.shape[0]:].\
            reshape(self.voxels[0, :, :, :].shape)

    def solve_ipopt_multi_measurement(self):
        # ######################## #
        #    Problem parameters    #
        # ######################## #
        self.t_ind = 30
        self.t_int = 1
        self.x_size = self.voxels[0, :].flatten().shape[0]
        # ######################## #
        #   Optimization problems  #
        # ######################## #
        self.method = "grad"
        # ######################## #
        #  Optimization variables  #
        # ######################## #
        self.y = self.data.electrode_rec[:, self.t_ind:self.t_ind+self.t_int]
        self.y_size = self.y.flatten().shape[0]
        self.w = ca.MX.sym("w", self.x_size * (self.t_int + 1) + self.y_size)
        self.ys = self.w[0:self.y_size].reshape((self.y.shape))
        self.xs = self.w[self.y_size:self.y_size +
                         self.x_size].reshape((self.x_size, 1))
        self.x = self.w[self.y_size +
                        self.x_size:].reshape((self.x_size, self.t_int))
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        dw = self.cmp_weight_matrix(fwd)
        self.fwd = np.dot(fwd, dw)
        self.sigma = 0
        # self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.f = 0
        self.g = []
        # bounds
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] -
                              ca.dot(self.fwd[i, :], self.x[:, ti]))
                self.lbg.append(0)
                self.ubg.append(0)
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)
        for j in range(self.x.nz[:].shape[0]):
            tmp = 0
            if j < (self.xs.shape[0]):
                for tj in range(self.t_int):
                    tmp += self.x.nz[:][tj*self.x_size+j]**2
                self.f += self.sigma*(self.xs[j])
                self.g.append(-self.xs[j]-tmp)
                self.g.append(-self.xs[j]+tmp)
                self.g.append(-self.xs[j])
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.ubg.append(0.)
                self.ubg.append(0.)
                self.ubg.append(0.)
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)
        self.lbx = ca.vertcat(*self.lbx)
        self.ubx = ca.vertcat(*self.ubx)
        # Initialize
        self.w0 = ca.vertcat(np.zeros(self.w.shape[0]))
        # Create NLP
        self.nlp = {"x": self.w, "f": self.f, "g": self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000,
                     # "ipopt.linear_solver": 'MA97',
                     "ipopt.hessian_approximation": "limited-memory"}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, self.opts)
        # Solve NLP
        self.args = {}
        self.args["x0"] = self.w0
        self.args["lbx"] = self.lbx
        self.args["ubx"] = self.ubx
        self.args["lbg"] = self.lbg
        self.args["ubg"] = self.ubg
        self.res = self.solver(**self.args)
        self.xres = self.res["x"].full()[self.y_size+self.xs.shape[0]:].\
            reshape((self.voxels[0, :, :, :].shape[0],
                     self.voxels[0, :, :, :].shape[1],
                     self.voxels[0, :, :, :].shape[2],
                     self.t_int))

    def solve_ipopt_mmv_2p(self):
        # ######################## #
        #    Problem parameters    #
        # ######################## #
        self.t_ind = 30
        self.t_int = 1
        self.x_size = self.voxels[0, :].flatten().shape[0]
        # ######################## #
        #   Optimization problems  #
        # ######################## #
        self.method = "grad"
        # ######################## #
        #  Optimization variables  #
        # ######################## #
        self.y = self.data.electrode_rec[:, self.t_ind:self.t_ind+self.t_int]
        self.y_size = self.y.flatten().shape[0]
        self.w = ca.MX.sym("w", self.x_size * (self.t_int + 1) + self.y_size)
        self.ys = self.w[0:self.y_size].reshape((self.y.shape))
        self.xs = self.w[self.y_size:self.y_size +
                         self.x_size].reshape((self.x_size, 1))
        self.x = self.w[self.y_size +
                        self.x_size:].reshape((self.x_size, self.t_int))
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        dw = self.cmp_weight_matrix(fwd)
        self.fwd = np.dot(fwd, dw)
        self.sigma = 0
        # self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.f = 0
        self.g = []
        # bounds
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] -
                              ca.dot(self.fwd[i, :], self.x[:, ti]))
                self.lbg.append(0)
                self.ubg.append(0)
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)
        for j in range(self.x.nz[:].shape[0]):
            tmp = 0
            if j < (self.xs.shape[0]):
                for tj in range(self.t_int):
                    tmp += self.x.nz[:][tj*self.x_size+j]**2
                self.f += self.sigma*(self.xs[j])
                self.g.append(-self.xs[j]-tmp)
                self.g.append(-self.xs[j]+tmp)
                self.g.append(-self.xs[j])
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.ubg.append(0)
                self.ubg.append(0)
                self.ubg.append(0)
                self.lbx.append(-ca.inf)
                self.ubx.append(ca.inf)
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)
        self.lbx = ca.vertcat(*self.lbx)
        self.ubx = ca.vertcat(*self.ubx)
        # Initialize
        self.w0 = ca.vertcat(np.zeros(self.w.shape[0]))
        # Create NLP
        self.nlp = {"x": self.w, "f": self.f, "g": self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000,
                     #"ipopt.linear_solver": 'MA97',
                     "ipopt.hessian_approximation": "limited-memory"}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, self.opts)
        # Solve NLP
        self.args = {}
        self.args["x0"] = self.w0
        self.args["lbx"] = self.lbx
        self.args["ubx"] = self.ubx
        self.args["lbg"] = self.lbg
        self.args["ubg"] = self.ubg
        self.res = self.solver(**self.args)
        self.xress = self.res["x"].full()[self.y_size+self.xs.shape[0]:].\
            reshape((self.voxels[0, :, :, :].shape[0],
                     self.voxels[0, :, :, :].shape[1],
                     self.voxels[0, :, :, :].shape[2],
                     self.t_int))

    def addGradientConstraints(self):
        self.grad = self.cmp_gradient()
        # ################## #
        # Objective function #
        # ################## #
        self.g = []
        # bounds
        self.lbg = []
        self.ubg = []
        for j in range(self.x.shape[0]):
            # gradient
            self.g.append(self.grad[j])
            self.lbg.append(0)
            self.ubg.append(5)
