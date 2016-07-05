
#! Callback
#! =====================
from casadi import *
from numpy import *

#! In this example, we will demonstrate callback functionality for Ipopt.
#! Note that you need the fix https://github.com/casadi/casadi/wiki/enableIpoptCallback before this works
#!
#! We start with constructing the rosenbrock problem
x=SX.sym("x")
y=SX.sym("y")

f = (1-x)**2+100*(y-x**2)**2
nlp={'x':vertcat(x,y), 'f':f,'g':x+y}
fcn = Function('f', [x, y], [f])
 
# #! Simple callback
# #! ===============
# #! First we demonstrate a callback that does nothing more than printing some information on the screen

# it = 0

# @pycallback
# def simplecallback(f):
#   global it
#   print "====Hey, I'm an iteration===="
#   print "X_OPT = ", f.getOutput("x")
#   print f.getStat("iteration")
#   it = it + 1
#   if it > 5:
#     print "5 Iterations, that is quite enough!"
#     return 1 # With this statement you can halt the iterations
#   else:
#     return 0

# opts = {}
# opts["iteration_callback"] = simplecallback
# opts["tol"] = 1e-8
# opts["max_iter"] = 20
# solver = nlpsol("solver", "ipopt", nlp, opts)
# solver.setInput([-10]*2,"lbx")
# solver.setInput([10]*2,"ubx")
# solver.setInput([-10],"lbg")
# solver.setInput([10],"ubg")
# solver.evaluate()

#! Matplotlib callback
#! ===================
#! Now let's do some useful visualisations
#! We create a callable python class, i.e. one that has a __call__ member

from pylab import figure, subplot, contourf, colorbar, draw, show, plot, title

import time
import matplotlib
matplotlib.interactive(True)

class MyCallback(Callback):
  def __init__(self, name, nx, ng, np, opts={}):
    Callback.__init__(self)

    self.nx = nx
    self.ng = ng
    self.np = np

    opts['input_scheme'] = nlpsol_out()
    opts['output_scheme'] = ['ret']

    figure(1)
    subplot(111)
    
    x_,y_ = mgrid[-1:1.5:0.01,-1:1.5:0.01]
    z_ = DM.zeros(x_.shape)
    
    for i in range(x_.shape[0]):
      for j in range(x_.shape[1]):
        z_[i,j] = fcn(x_[i,j],y_[i,j])
    contourf(x_,y_,z_)
    colorbar()
    title('Iterations of Rosenbrock')
    draw()
    
    self.x_sols = []
    self.y_sols = []

    # Initialize internal objects
    self.construct(name, opts)

  def get_n_in(self): return nlpsol_n_out()
  def get_n_out(self): return 1


  def get_sparsity_in(self, i):
    n = nlpsol_out(i)
    if n=='f':
      return Sparsity. scalar()
    elif n in ('x', 'lam_x'):
      return Sparsity.dense(self.nx)
    elif n in ('g', 'lam_g'):
      return Sparsity.dense(self.ng)
    else:
      return Sparsity(0,0)
  def eval(self, arg):
    print 'hello'
    # Create dictionary
    darg = {}
    for (i,s) in enumerate(nlpsol_out()): darg[s] = arg[i]

    sol = darg['x']
    self.x_sols.append(float(sol[0]))
    self.y_sols.append(float(sol[1]))
    subplot(111)
    if hasattr(self,'lines'):
      self.lines[0].set_xdata(self.x_sols)
      self.lines[0].set_ydata(self.y_sols)
    else:
      self.lines = plot(self.x_sols,self.y_sols,'or-')

    draw()
    time.sleep(0.25)

    return [0]

mycallback = MyCallback('mycallback', 2, 1, 0)
opts = {}
opts['iteration_callback'] = mycallback
opts['iteration_callback_step'] = 1
opts['ipopt.tol'] = 1e-8
opts['ipopt.max_iter'] = 50
solver = nlpsol('solver', 'ipopt', nlp, opts)
sol = solver(lbx=-10, ubx=10, lbg=-10, ubg=10)

#! By setting matplotlib interactivity off, we can inspect the figure at ease
matplotlib.interactive(False)
show()