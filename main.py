#Simulation of fourth order heat equation 

import numpy as np
import matplotlib.pyplot as plt
from classes import Grid,Quadrature,Basis
from StationaryProblem import StationaryProblem
from timeStepping import timeStep
import math

##Grid Setup & Selection of basis function
grid = Grid(-np.pi,np.pi,100)             
quadrature = Quadrature(3)
basis = Basis(3)
g = lambda x: 0
a = lambda x: 0.001  #Coefficient for diffusion and Bi-harmonic equation
pde = StationaryProblem(grid,basis,quadrature,lambda x: True,g)
#pde.addDiffusion(a)
pde.addFourth(a)
pde.addM()

##Initial Conditions
#Reference functions for curve 
#f = lambda x: 2*np.exp((-4*(x)**2))
#f = lambda x: (-5*(x)**3)+(10*(x)**2)+2*x+2
#f = lambda x:-np.arctan(x)
f = lambda x : np.sin(x)
U0 = grid.points[:,0]
U0 = f(U0)

#Time Stepping 
time = timeStep(pde,U0,0.2, step=0.01)
time.solve()



