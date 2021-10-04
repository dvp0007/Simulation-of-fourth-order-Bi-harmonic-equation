import numpy as np
import matplotlib.pyplot as plt
from classes import Grid,Quadrature,Basis



class StationaryProblem():
    def __init__(self, grid, basis, quadrature, dirichletLocations = (lambda x: True), dirichletValues = (lambda x: 0.0)):
      self.grid = grid
      self.basis = basis
      self.quadrature = quadrature
     
      # Initialize 
      self.systemMatrix = np.zeros((np.shape(self.grid.points)[0],np.shape(self.grid.points)[0]))
      self.systemVector = np.zeros(np.shape(self.grid.points)[0])
      self.solution = None

      self.matrixM = np.zeros((np.shape(self.systemMatrix)[0],np.shape(self.systemMatrix)[0]))
      
      # Boundary Conditions
      self.dirichletDOFs = self.grid.getBoundaryIndices(dirichletLocations)
      self.allDOFs = np.arange(np.shape(self.grid.points)[0])
      self.freeDOFs = np.setdiff1d(np.arange(np.shape(self.grid.points)[0]),self.dirichletDOFs)    
      self.assembleBoundaryConditions(dirichletValues)
      
      # Precompute data that is needed all the time.
      self.xkHat,self.wkHat = self.quadrature.getPointsAndWeights()
      self.xkTrafo = self.grid.evalReferenceMap(self.xkHat)
      self.dets = np.abs(self.grid.getDeterminants())
      self.invJac = self.grid.getInverseJacobians()  
      self.phi = self.basis.evalPhi(self.xkHat)
      self.gradPhi = self.basis.evalGradPhi(self.xkHat)
      self.doubleGrad = self.basis.doubleGrad(self.xkHat)
          

    def assembleBoundaryConditions(self, dirichletValues):
      self.systemMatrix[self.dirichletDOFs,self.dirichletDOFs] = 1.0
      self.matrixM[self.dirichletDOFs,self.dirichletDOFs] = 1.0
#       self.systemVector[self.dirichletDOFs] = dirichletValues(np.array([40,3]))
      self.systemVector[self.dirichletDOFs] = dirichletValues(self.grid.points[self.dirichletDOFs,0])
     

    def addSource(self, f):
      for i in self.freeDOFs:  
          supp,localInd = self.grid.evalDOFMap(i)
          for T,loc_i in zip(supp,localInd):
              self.systemVector[i] += self.dets[T,0]*np.sum(self.phi[:,loc_i] * self.wkHat * f(self.xkTrafo[T,0]))
        
      
    def addDiffusion(self, a):
      for i in self.freeDOFs:
          for j in self.allDOFs:        
              supp_I,localIndices_I = self.grid.evalDOFMap(i)
              supp_J,localIndices_J = self.grid.evalDOFMap(j)
              
              supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
              localIndices_I = localIndices_I[tmpLocalIndices_I]
              localIndices_J = localIndices_J[tmpLocalIndices_J]
              
              for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
                  for k in range(self.xkHat.shape[0]):
                      self.systemMatrix[i,j] += a(self.xkTrafo[T,k,0]) * self.dets[T,0] * self.wkHat[k] * np.dot(np.dot(self.invJac[T,:],self.gradPhi[k,loc_j,:]),np.dot(self.invJac[T,:],self.gradPhi[k,loc_i,:]))

    def addFourth(self,a):
        for i in self.freeDOFs:
            for j in self.allDOFs:        
                supp_I,localIndices_I = self.grid.evalDOFMap(i)
                supp_J,localIndices_J = self.grid.evalDOFMap(j)
                
                supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
                localIndices_I = localIndices_I[tmpLocalIndices_I]
                localIndices_J = localIndices_J[tmpLocalIndices_J]
                
                for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
                    for k in range(self.xkHat.shape[0]):
                       # self.systemMatrix[i,j] += self.wkHat[k] * np.dot(np.dot(self.invJac[T,:],self.doubleGrad[k,loc_j,:]),np.dot(self.invJac[T,:],self.doubleGrad[k,loc_i,:]))
                        self.systemMatrix[i,j] += a(self.dets[T,0]) * self.wkHat[k] * np.dot(np.dot(self.invJac[T,:],self.doubleGrad[k,loc_j,:]),np.dot(self.invJac[T,:],self.doubleGrad[k,loc_i,:]))


    def addConvection(self, b):
      self.solution = None
      
      for i in self.freeDOFs:
          for j in self.allDOFs:        
              supp_I,localIndices_I = self.grid.evalDOFMap(i)
              supp_J,localIndices_J = self.grid.evalDOFMap(j)
              
              supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
              localIndices_I = localIndices_I[tmpLocalIndices_I]
              localIndices_J = localIndices_J[tmpLocalIndices_J]
              
              for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
                  for k in range(self.xkHat.shape[0]):
                      self.systemMatrix[i,j] +=  self.dets[T,0] * self.wkHat[k] * np.dot(b(self.xkTrafo[T,k,0]),np.dot(self.invJac[T,:],self.gradPhi[k,loc_j,:])) * self.phi[k,loc_i,]


    def addReaction(self, c):
      self.solution = None
      
      for i in self.freeDOFs:
          for j in self.allDOFs:        
              supp_I,localIndices_I = self.grid.evalDOFMap(i)
              supp_J,localIndices_J = self.grid.evalDOFMap(j)
              
              supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
              localIndices_I = localIndices_I[tmpLocalIndices_I]
              localIndices_J = localIndices_J[tmpLocalIndices_J]
              
              for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
                  for k in range(self.xkHat.shape[0]):
                      self.systemMatrix[i,j] +=  self.dets[T,0] * self.wkHat[k] * c(self.xkTrafo[T,k,0]) * self.phi[k,loc_j] * self.phi[k,loc_i]
     
                      
    def addM(self, c = lambda x: 1):
      self.solution = None
      
      for i in self.freeDOFs:
          for j in self.allDOFs:        
              supp_I,localIndices_I = self.grid.evalDOFMap(i)
              supp_J,localIndices_J = self.grid.evalDOFMap(j)
              
              supp_IJ,tmpLocalIndices_I,tmpLocalIndices_J = np.intersect1d(supp_I,supp_J,assume_unique=False,return_indices=True)
              localIndices_I = localIndices_I[tmpLocalIndices_I]
              localIndices_J = localIndices_J[tmpLocalIndices_J]
              
              for T,loc_i,loc_j in zip(supp_IJ,localIndices_I,localIndices_J):
                  for k in range(self.xkHat.shape[0]):
                      self.matrixM[i,j] +=  self.dets[T,0] * self.wkHat[k] * c(self.xkTrafo[T,k,0]) * self.phi[k,loc_j] * self.phi[k,loc_i]

    def addNuemann(self):
      for i in [0, len(self.systemVector)]:
            supp,localInd = self.grid.evalDOFMap(i)
            for T,loc_i in zip(supp,localInd):  
                self.systemVector[i] += self.dets[T,0]*np.sum(self.phi[:,loc_i] * self.wkHat)
            

        
    def solve(self):
      self.solution = np.linalg.solve(self.systemMatrix , self.systemVector)

    def show(self):
      self.grid.plotDOFVector(self.solution)