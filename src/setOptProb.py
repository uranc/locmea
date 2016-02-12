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
        self.y = ca.SX(self.data.electrode_rec[:, self.t_ind:self.t_ind+self.t_int])
        self.x = ca.SX.sym('x', self.voxels[0, :].flatten().shape[0])
        # self.alpha = ca.MX.sym('alpha', self.x.shape[0])
        self.fwd = ca.SX(self.cmp_fwd_matrix(self.electrode_pos, self.voxels))
        # self.grad = self.cmp_gradient()
        self.sigma = 1
        # Objective
        self.f = 0
        # self.fvector = (self.y-ca.mul(self.fwd,self.x))**2
        # self.f = []
        #for i in range(self.y.shape[0]):
        #    self.tmp = []
        #    for n in range(self.fwd.shape[1]):
        #        self.tmp =sum(self.fwd[i, n]*self.x[n])
        #    self.f += sum((self.y[i] - sum(self.tmp))**2)
        self.f += sum([(self.y[i] - sum([self.fwd[i, n]*self.x[n] for n in
                      range(self.fwd.shape[1])])**2) for i in
            range(self.y.shape[0])])
        #self.f.shape
        # assert isinstance(time, ca.SX)
        # Constraint
        #self.g = []
        # self.g = ca.vertcat([self.g, self.grad])
        #self.g = ca.vertcat([self.g, (self.x**2-self.alpha**2)])
        # Bounds
        #self.lbg = []
        #self.ubg = []
        self.lbx = np.ones(self.x.shape[0])*-100.
        self.ubx = np.ones(self.x.shape[0])*100.
        #self.lbx = ca.vertcat([self.lbx, np.ones(self.x.shape[0])*-100.])
        #self.ubx = ca.vertcat([self.ubx, np.ones(self.x.shape[0])*100.])
        #self.lbg = ca.vertcat([self.lbg, np.ones(self.x.shape[0])*0])
        #self.ubg = ca.vertcat([self.ubg, np.ones(self.g.shape[0])*0])
        #self.x = ca.vertcat([self.alpha, self.x])
        # Initialize
        self.x0 = np.ones(self.x.shape[0])*0.
        # Create NLP
        print self.f
        self.nlp = ca.SXFunction("nlp", ca.nlpIn(x=self.x),
                                 ca.nlpOut(f=self.f))  #, g=self.g))
        # NLP solver options
        self.opts = {"max_iter": 10000}
        # "iteration_callback_step": self.plotUpdateSteps}
        # Create solver
        print "hallo"
        self.solver = ca.NlpSolver("solver", "ipopt", self.nlp)  #, self.opts)
        # Solve NLP
        self.solver.setInput(self.x0, "x0")  # initial guess
        self.solver.setInput(self.lbx, "lbx")  # lower boundary on x
        self.solver.setInput(self.ubx, "ubx")  # upper boundary on x
        #self.solver.setInput(self.lbg, "lbg")  # boundary on g
        #self.solver.setInput(self.ubg, "ubg")  # boundary on g
        # self.solver.evaluate()
        # self.solution_x = self.solver.getOutput("x")

    @ca.pycallback
    def plot_updates(self, f):
        """
        Gets inter-optimization plots
        """
        print "callback here"
        return 0

    def cmp_dx(self, i, j, k, t, h=1):
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
            dx = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif i == ni - 1:
            # dx = 0
            dx = 0
        elif i == ni - 2:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            ind1 = np.ravel_multi_index((i + 1, j, k), vx.shape)
            ind_1 = np.ravel_multi_index((i - 1, j, k), vx.shape)
            ind2 = np.ravel_multi_index((i + 2, j, k), vx.shape)
            ind_2 = np.ravel_multi_index((i - 2, j, k), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dx = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dx

    def cmp_dy(self, i, j, k, t, h=1):
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
            dy = (8*(x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif j == nj - 1:
            # dy = 0
            dy = 0
        elif j == nj - 2:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
            # dy = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dy = (8*(x[ind1, t] - x[ind_1, t]) - 1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            ind1 = np.ravel_multi_index((i, j + 1, k), vx.shape)
            ind_1 = np.ravel_multi_index((i, j - 1, k), vx.shape)
            ind2 = np.ravel_multi_index((i, j + 2, k), vx.shape)
            ind_2 = np.ravel_multi_index((i, j - 2, k), vx.shape)
            # dy = 8(x1-x_1)-1(x2-x_2)
            dy = (8*(x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dy

    def cmp_dz(self, i, j, k, t, h=1):
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
            dz = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind0, t]))/(12*h)
        elif k == nk - 1:
            # dx = 0
            dz = 0
        elif k == nk - 2:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x0-x_2) x0 instead of x2 (mirror)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind0, t] - x[ind_2, t]))/(12*h)
        else:
            ind1 = np.ravel_multi_index((i, j, k + 1), vx.shape)
            ind_1 = np.ravel_multi_index((i, j, k - 1), vx.shape)
            ind2 = np.ravel_multi_index((i, j, k + 2), vx.shape)
            ind_2 = np.ravel_multi_index((i, j, k - 2), vx.shape)
            # dx = 8(x1-x_1)-1(x2-x_2)
            dz = (8 * (x[ind1, t] - x[ind_1, t]) - 1*(x[ind2, t] - x[ind_2, t]))/(12*h)
        return dz

    def cmp_gradient(self, flag_tmp_smooth=False, h=1, p_diff='2'):
        """
        <F7>cmp_gradient
        """
        # initials
        x = self.x
        vx, vy, vz = self.voxels
        ni, nj, nk = vx.shape
        nv = ni * nj * nk
        if x.size() != nv:
            nt = nv / x.size()
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
                                ca.sumRows(self.cmp_dx(i, j, k, t ,h))**2 +
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

    def initial_guess(self):
        """
        ##############
        initial guess
        ##############
        """
    def set_constraints(self):
        """
        ##########
        set some constraints
        ##########
        """
        
    def minimization_module(self):
        """
        ##############
        set optimization problem
        ##############
        """
