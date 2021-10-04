import numpy as np
import matplotlib.pyplot as plt
from classes import Grid,Quadrature,Basis
from StationaryProblem import StationaryProblem
from matplotlib.animation import FuncAnimation


class timeStep():
    def __init__(self, StationaryProblem,intialCondition,time=1,step=0.1):
        self.pde = StationaryProblem
        self.U0 = intialCondition
        self.t = time
        self.dt = step


        self.timeGrid = np.arange(0,self.t,self.dt)
        # self.matrixM = np.zeros((np.shape(self.pde.systemMatrix)[0],np.shape(self.pde.systemMatrix)[0]))

    
    # def creatMatrixM(self, c = lambda x: 1):
    #   self.solution = None
      
    #   for i in self.freeDOFs:
    #       for j in self.allDOFs:        
    #           supp_I,localIndices_I = self.grid.evalDOFMap(i)
    #           supp_J,localIndices_J = self.grid.evalDOFMap(j)
              
    #           supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
    #           localIndices_I = localIndices_I[tmpLocalIndices_I]
    #           localIndices_J = localIndices_J[tmpLocalIndices_J]
              
    #           for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
    #               for k in range(self.xkHat.shape[0]):
    #                   self.matrixM[i,j] +=  self.dets[T,0] * self.wkHat[k] * c(self.xkTrafo[T,k,0]) * self.phi[k,loc_j] * self.phi[k,loc_i]


    def plot(self,vec):
        plt.ion()
        plt.figure(1)
        plt.plot(self.pde.grid.points[:,0],vec)
        plt.ylim(-3,3)
        #plt.xlim(-0.5,0.5)
        plt.show()
        plt.pause(2)
        plt.cla()

    def solve(self, scheme = 'Eular Forward'):
        A = (np.dot(self.pde.matrixM,(1/self.dt)))+self.pde.systemMatrix
        b = self.pde.systemVector

        if scheme == 'Eular Forward':
            for i in range(1,len(self.timeGrid)):
                b1 = b + np.dot(np.dot(self.pde.matrixM,self.U0),1/self.dt) 
                Ui = np.linalg.solve(A,b1)
                # print(Ui)np
                self.plot(self.U0)
                self.U0 = Ui
            
