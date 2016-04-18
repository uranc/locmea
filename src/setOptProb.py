"""
Create attributes for the optimization problem
"""
# Author: Cem Uran <cem.uran@uranus.uni-freiburg.de>
# License:
from setInvProb import data_out
import casadi as ca
import numpy as np
from casadi.tools import struct_symMX, entry, repeated

# @ca.pycallback
# class MyCallback:
#     def __init__(self):
#         print 'hellooo'
#     def __call__(self,f,*args):
#         sol = f.getOutput("x")


class opt_out(data_out):
    """
    Class for the optimization problem
    Child of inverse problem
    """
    def __init__(self, *args, **kwargs):
        # ######################## #
        # Options for optimization #
        # ######################## #
        data_out.__init__(self, *args, **kwargs)
        self.opt_opt = {'solver': 'ipopt', 'datafile_name': 'output_file.dat', 'callback_steps': 1,
                        'method': 'grad','t_ind': 30, 't_int': 1, 'flag_depthweighted': True, 'sigma': 0.1}
        self.opt_opt.update(kwargs)
        self.solver = self.opt_opt['solver']
        self.datafile_name = self.opt_opt['datafile_name']
        self.callback_steps = self.opt_opt['callback_steps']
        self.t_ind = self.opt_opt['t_ind']
        self.t_int = self.opt_opt['t_int']
        self.sigma = self.opt_opt['sigma']
        self.method = self.opt_opt['method']
        self.flag_depthweighted = self.opt_opt['flag_depthweighted']
        # ######################## #
        #     Problem  setup       #
        # ######################## #
        print self.opt_opt
        self.y = self.data.electrode_rec[:, self.t_ind:self.t_ind+self.t_int]
        self.y_size = self.y.flatten().shape[0]
        self.x_size = self.voxels[0, :].flatten().shape[0]
        fwd = self.cmp_fwd_matrix(self.electrode_pos, self.voxels)
        if self.flag_depthweighted:
            dw = self.cmp_weight_matrix(fwd)
            self.fwd = np.dot(fwd, dw)
        else:
            self.fwd = ca.MX(fwd)

    @ca.pycallback
    def alternating_optimization(self, f):
        '''
        hello there
        '''
        print 'hello'
        # a_tmp = f.getOutput("x")[:self.x_size*self.t_int].reshape(self.s.shape)
        # for px in range(self.a.shape[0]):
        #     f.getOutput("x")[:self.x_size*self.t_int]
        #     tmp_res = self.optimize_waveform(a_tmp[px, :])
        return 0

    def set_optimization_variables_slack(self):
        """
        Variables for the lifted version
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

    def minimize_function(self):
        """
        Function where the NLP is initialized and solved
        Regardless of the model made
        """
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)
        # Initialize at 0
        self.w0 = self.w(0)
        # Create NLP
        self.nlp = {"x": self.w, "f": self.f, "g": self.g}
        # NLP solver options
        self.opts = {"ipopt.max_iter": 100000,
                     # "compute_red_hessian": "yes",
                     # "ipopt.linear_solver": 'MA97',
                     #"iteration_callback": MyCallback(),
                     "iteration_callback": self.alternating_optimization,
                     "iteration_callback_step": self.callback_steps,
                     "ipopt.hessian_approximation": "limited-memory"}
        # Create solver
        print "Initializing the solver"
        self.solver = ca.nlpsol("solver", "ipopt", self.nlp, self.opts)
        #self.solver = ca.qpsol("solver", "qpoases", self.nlp,{'sparse':True})
        # Solve NLP
        self.args = {}
        self.args["x0"] = self.w0
        self.args["lbx"] = self.lbx
        self.args["ubx"] = self.ubx
        self.args["lbg"] = self.lbg
        self.args["ubg"] = self.ubg
        self.res = self.solver(**self.args)

    def add_data_costs_constraints_slack(self):
        """
        Computes objective function f
        With lifting variable ys constraints
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] -
                              ca.dot(self.fwd[i, :], self.x[:, ti]))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_l1_costs_constraints_slack(self):
        """
        add slack l1 constraints with lifting variables
        """
        for j in range(self.w['xs'].shape[0]):
            tmp = 0
            for tj in range(self.t_int):
                tmp += self.w['x'][j,tj]**2
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

    def cmp_dx(self, smooth_entity, i, j, k, t, h=1.):
        """
        cmp_dx
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

    def cmp_dy(self, smooth_entity, i, j, k, t, h=1.):
        """
        cmp_dy
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

    def cmp_dz(self, smooth_entity, i, j, k, t, h=1.):
        """
        cmp_dz
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

    def cmp_gradient(self, smooth_entity, flag_tmp_smooth=False, h=1., flag_second=True):
        """
        <F7>cmp_gradient
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
        print "Time point(s): ", nt
        # loop over the voxels
        for t in range(nt):
            for i in range(ni):
                for j in range(nj):
                    for k in range(nk):
                        ind = np.ravel_multi_index((i, j, k), vx.shape)
                        if ind == 0:
                            grad_mtr = [sum([self.cmp_dx(smooth_entity, i, j, k, t, h)]),
                                        sum([self.cmp_dy(smooth_entity, i, j, k, t, h)]),
                                        sum([self.cmp_dz(smooth_entity, i, j, k, t, h)])]
                        else:
                            grad_mtr = ca.horzcat(grad_mtr, 
                                ca.vertcat(sum([self.cmp_dx(smooth_entity, i, j, k, t, h)]),
                                sum([self.cmp_dy(smooth_entity, i, j, k, t, h)]),
                                sum([self.cmp_dz(smooth_entity, i, j, k, t, h)])))
        if flag_tmp_smooth:
            # compute temporal gradient
            print "Temporal smoothness enforced."
        return grad_mtr

    def optimize_waveform(self, x):
        """
        fit waveform to a bimodal alpha function
        """
        srate = self.data.srate
        #fit_data = self.data.cell_csd[0, 30:]
        fit_data = x
        tlin = np.linspace(0, (fit_data.shape[0]-1)/srate, fit_data.shape[0])
        t = ca.MX.sym("t")
        t1 = ca.MX.sym("t1")
        t2 = ca.MX.sym("t2")
        a = ca.MX.sym("a")
        r = ca.vertcat(t1, t2, a)
        f = (ca.exp(-t*t1)*t*t1*t1 - ca.exp(-t*t2)*t*t2*t2)*a
        F = ca.Function("F", [r, t], [f])
        Y = [(fit_data[i]-F(r, tlin[i]))**2
             for i in range(fit_data.shape[0])]
        nlp_root = {"x": r, "f": sum(Y)}
        root_solver = ca.nlpsol("solver", "ipopt", nlp_root)
        r0 = [1.e3, 2.e3, 1.]
        args = {}
        args["x0"] = r0
        args["lbx"] = ca.vertcat([-ca.inf, -ca.inf, -ca.inf])
        args["ubx"] = ca.vertcat([ca.inf, ca.inf, ca.inf])
        res = root_solver(**args)
        return [F(res["x"], tlin[i]) for i in range(fit_data.shape[0])]

    def solve_ipopt_multi_measurement_slack(self):
        """
        MMV L1 
        """
        self.set_optimization_variables_slack()
        self.add_data_costs_constraints_slack()
        self.add_l1_costs_constraints_slack()
        #self.add_gradient_costs_constraints()
        self.minimize_function()

    def set_optimization_variables_2p(self):
        """
        x is divided into negative and positive elements
        this function overwrites the initialized optimization variables
        """
        self.w = struct_symMX([entry("xs_pos", shape=(self.x_size*2,self.t_int)),
                               entry("xs_neg", shape=(self.x_size*2,self.t_int)),
                               entry("ys", shape=(self.y.shape))])
        self.xs_pos, self.xs_neg, self.ys = self.w[...]
        self.g = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)
        print "New measurement matrix"
        self.fwd = ca.horzcat(self.fwd, self.fwd)

    def add_data_costs_constraints_2p(self):
        """
        Computes objective function f
        With lifting variable ys constraints
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] -
                              ca.dot(self.fwd[i, :], (self.xs_pos[:, ti]-self.xs_neg[:, ti]).T))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_l1_costs_constraints_2p(self):
        """
        add slack l1 constraints with lifting variables
        """
        for j in range(self.xs_pos.shape[0]):
            tmp_pos = 0
            tmp_neg = 0
            for tj in range(self.t_int):
                if j < self.xs_pos.shape[0]/2:
                    tmp_pos += self.xs_pos[j,tj]**2
                    tmp_neg += self.xs_neg[j,tj]**2
                self.g.append(-self.xs_pos[j,tj])
                self.g.append(-self.xs_neg[j,tj])
                self.lbg.append(-ca.inf)
                self.lbg.append(-ca.inf)
                self.ubg.append(0.)
                self.ubg.append(0.)
            self.f += self.sigma*(tmp_pos+tmp_neg)

    def solve_ipopt_multi_measurement_2p(self):
        """
        Reform source space x as the difference of x+ - x-
        """
        self.set_optimization_variables_2p()
        self.add_data_costs_constraints_2p()
        self.add_l1_costs_constraints_2p()
        self.minimize_function()

    def set_optimization_variables_thesis(self):
        """
        thesis implementation
        """
        self.w = struct_symMX([entry("a", shape=(self.x_size,self.t_int)),
                               entry("m", shape=(self.x_size)),
                               entry("s", shape=(self.x_size,3)),
                               entry("ys", shape=(self.y.shape))])
        self.a, self.m, self.s, self.ys = self.w[...]
        self.g = []
        self.lbg = []
        self.ubg = []
        self.f = 0
        self.lbx = self.w(-ca.inf)
        self.ubx = self.w(ca.inf)
        self.lbx['m'] = 0
        self.ubx['m'] = 1

    def add_data_costs_constraints_thesis(self):
        """
        Computes objective function f
        With lifting variable ys constraints
        """
        for i in range(self.y.shape[0]):
            for ti in range(self.t_int):
                self.f += (self.y[i, ti] - self.ys[i, ti])**2
                self.g.append(self.ys[i, ti] -
                              ca.dot(self.fwd[i, :], (self.a[:, ti]).T))
                self.lbg.append(0)
                self.ubg.append(0)

    def add_l1_costs_constraints_thesis(self):
        """
        add slack l1 constraints with lifting variables
        """
        for j in range(self.m.shape[0]):
            self.f += self.sigma*(self.m[j])

    def add_background_costs_constraints_thesis(self):
        """
        add background constraints with lifting variables
        """
        for b in range(self.m.shape[0]):
            tmp = 0
            for tb in range(self.t_int):
                    tmp += self.a[b,tb]**2
            self.g.append(tmp*self.m[b])
            self.lbg.append(0)
            self.ubg.append(0)

    def add_tv_mask_costs_constraints_thesis(self):
        """
        add smoothness constraints with lifting variables
        """
        grad_m = self.cmp_gradient(self.m)
        for cm in range(self.m.shape[0]):
            self.g.append(ca.sqrt(ca.dot(grad_m[:,cm],grad_m[:,cm])))
            self.lbg.append(0)
            self.ubg.append(10)

    def add_smoothness_costs_constraints_thesis(self):
        """
        add smoothness constraints with lifting variables
        """ 
        for tb in range(self.t_int):
            tmp = self.cmp_gradient(self.a[:,tb])
            for b in range(self.x_size):
                self.g.append(ca.dot(self.s[b,:].T,tmp[:,b])*self.m[b])
                self.lbg.append(0)
                self.ubg.append(10)

    def add_s_magnitude_costs_constraints_thesis(self):
        """
        add smoothness constraints with lifting variables
        """
        for b in range(self.x_size):
            self.g.append(ca.dot(self.s[b,:],self.s[b,:])-1)
            self.lbg.append(0)
            self.ubg.append(0)

    def add_s_smooth_costs_constraints_thesis(self):
        """
        add smoothness constraints with lifting variables
        """
        grad = []
        grad_x = self.cmp_gradient(self.s[:,0])
        grad_y = self.cmp_gradient(self.s[:,1])
        grad_z = self.cmp_gradient(self.s[:,2])
        for b in range(self.s.shape[0]):
            self.g.append(ca.sqrt(ca.dot(grad_x[:, b],grad_x[:, b])+
                                  ca.dot(grad_y[:, b],grad_y[:, b])+
                                  ca.dot(grad_z[:, b],grad_z[:, b])))
            self.lbg.append(0)
            self.ubg.append(10)

    def solve_ipopt_multi_measurement_thesis(self):
        """
        Reform source space x as the difference of x+ - x-
        """
        self.set_optimization_variables_thesis()
        self.add_data_costs_constraints_thesis()
        self.add_l1_costs_constraints_thesis()
        self.add_background_costs_constraints_thesis()
        self.add_tv_mask_costs_constraints_thesis()
        self.add_smoothness_costs_constraints_thesis()
        self.add_s_magnitude_costs_constraints_thesis()
        self.add_s_smooth_costs_constraints_thesis()
        self.minimize_function()

    def initialization(self):
        """
        initialization for the optimization problem
        """
        self.g.append()