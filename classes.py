import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.tri as triang



class Grid():
    def __init__(self,xlow,xhigh,Nx):
        
        # store the variables
        self.xlow = xlow
        self.xhigh = xhigh
        self.Nx = Nx
        
        # creat the grid
        self.points = None
        self.cells = None
        self.creatGrid()
        
        # create the determinant and the Inverse of the Jacobian 
        self.det = None
        self.invJac = None
        self.computeTrafoInformation()        
        
    def creatGrid(self):
        # vectors for division
        self.points = np.linspace(self.xlow,self.xhigh,self.Nx)
        self.points = self.points.reshape(self.Nx,1)

    
        # number of 1D elements
        nr_x = self.Nx-1

        # elements index
        self.cells = np.stack((np.arange(nr_x), np.arange(1,self.Nx)), axis=-1)


        
    def computeTrafoInformation(self):
    
        verts = self.points[self.cells]
        verts = np.swapaxes(verts,2,1)

        # tranforamtion matrix "coffienat of x in maping function"
        trafoMat = np.stack((verts[:,:,1]-verts[:,:,0]))

        self.det = trafoMat

        # inverse 
        self.invJac = np.divide(1,trafoMat)
    
    def evalReferenceMap(self,xHat):
        # get vertices in format nEx2x3
        verts = self.points[self.cells]
        verts = np.swapaxes(verts,2,1)

        # define reference matrix and vector
        trafoVec = verts[:,:,0]
        trafoMat = np.stack((verts[:,:,1]-verts[:,:,0]))
            
        if len(xHat.shape)>1:
          trafoX =  trafoMat[:,None,None,0]*xHat[None,:] + trafoVec[:,None]  
        else:
            trafoX = np.dot(trafoMat,xHat) + trafoVec
        
        return trafoX        
    
    def evalDOFMap(self,globalInd):
        supp,localInd = np.where(self.cells == globalInd)
        return supp,localInd        
     
    def getDivisions(self):
      return self.Nx

    def getDeterminants(self):
        return self.det

    def getInverseJacobians(self):
      return self.invJac

    def getPoints(self):
        return self.points

    def getInnerIndices(self):
        bounds_low = np.array([self.xlow])
        # bounds_high = np.array([self.xhigh])

        # return np.where(np.minimum(np.min(np.abs(self.points-bounds_low),axis=1),np.min(np.abs(self.points-bounds_high),axis=1)) >1e-6)[0]
        return np.where(np.min(np.abs(self.points-bounds_low),axis=1) >1e-6)[0]


    def getBoundaryIndices(self, locator = (lambda x: True)):
        innerInd = self.getInnerIndices()
        allInd = np.arange(np.shape(self.points)[0])

        boundaryInd = np.setdiff1d(allInd,innerInd)

        return boundaryInd[locator(self.points[boundaryInd,0])]
    
    def plotDOFVector(self, vec):
        
        fig = plt.figure()

        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([self.xlow, self.xhigh])
        ax.plot(self.points[:,0],vec)
        
        plt.show()
    
    
    def show(self):
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        ax.plot(self.points,np.zeros_like(self.points),'-o')

        plt.show()     








class Basis():
    def __init__(self, order=1):
        self.order = order
    
    def evalPhi(self,xHat):
        if self.order == 1:
            N = np.shape(xHat)[0]
            phi = np.zeros((N,2))
            phi[:,0] = 1 - xHat[:,0]
            phi[:,1] = xHat[:,0]
            return phi
        if self.order == 2:
            N = np.shape(xHat)[0]
            phi = np.zeros((N,2))
            phi[:,0] = 1 - xHat[:,0]
            phi[:,1] = xHat[:,0]
            return phi
        if self.order == 3:
            N = np.shape(xHat)[0]
            phi = np.zeros((N,4))
            phi[:,0] = 2*xHat[:,0]**3 - 3*xHat[:,0]**2 + 1
            phi[:,1] = xHat[:,0]**3 - 2*xHat[:,0]**2 + xHat[:,0]
            phi[:,2] = -2*xHat[:,0]**3 + 3*xHat[:,0]**2
            phi[:,3] = xHat[:,0]**3 - xHat[:,0]**2
            return phi

    
    def evalGradPhi(self,xHat):
        if self.order == 1:
            N = np.shape(xHat)[0]
            gradPhi = np.zeros((N,2,1))
            gradPhi[:,0,0] = -1
            gradPhi[:,1,0] = 1  
            return gradPhi
        if self.order == 2:
            N = np.shape(xHat)[0]
            gradPhi = np.zeros((N,2,1))
            gradPhi[:,0,0] = -1
            gradPhi[:,1,0] = 1  
            return gradPhi    
        if self.order == 3:
            N = np.shape(xHat)[0]
            gradPhi = np.zeros((N,4,1))
            gradPhi[:,0,0] = 6*xHat[:,0]**2 - 6*xHat[:,0]
            gradPhi[:,1,0] = 3*xHat[:,0]**2 - 4*xHat[:,0] + 1
            gradPhi[:,2,0] = -6*xHat[:,0]**2 + 6*xHat[:,0]
            gradPhi[:,3,0] = 3*xHat[:,0]**2 - 2*xHat[:,0]  
            return gradPhi    

    def doubleGrad(self,xHat):
        N = np.shape(xHat)[0]
        doubleGrad = np.zeros((N,4,1))
        doubleGrad[:,0,0] = 12*xHat[:,0] - 6
        doubleGrad[:,1,0] = 6*xHat[:,0] - 4
        doubleGrad[:,2,0] = -12*xHat[:,0] + 6
        doubleGrad[:,3,0] = 6*xHat[:,0] - 2  
        return doubleGrad 


    def plotPhi(self, index):
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        xHat = np.array([[0.0],[0.32], [0.65], [1.0]])
        phi = self.evalPhi(xHat)
        f = interp1d(xHat[:,0],phi[:,index], kind='cubic')

        plt.plot(xHat[:,0],f(xHat[:,0]))
    
        # plt.plot(xHat[:,0],phi[:,index])
        plt.show()




class Quadrature:
    def __init__(self, order):
        if order == 1:
            self.weights = np.array([2])
            self.points = np.array([[0]])
        if order == 2:
            self.weights = np.array([1, 1])
            self.points = np.array([[np.sqrt(1/3)],[-np.sqrt(1/3)]])     
        else:
            self.weights = np.array([(18+np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36, (18-np.sqrt(30))/36])
            self.points = np.array([[np.sqrt(3/7-2/7*np.sqrt(6/5))],[-np.sqrt(3/7-2/7*np.sqrt(6/5))],[np.sqrt(3/7+2/7*np.sqrt(6/5))],[-np.sqrt(3/7+2/7*np.sqrt(6/5))]]) 
            if order > 4:
                print("Requested order " + str(order) + " is not implemented. Returning order 4 quadrature.")

    def integrateFunction(self, fun):
        #fun - scalar lambda function defined in 2D, i.e. fun = lambda x,y: ....
        return np.sum(fun(self.points[:,0]) * self.weights)    

    def getPointsAndWeights(self):
        return self.points,self.weights



print(Grid(0,1,10).getBoundaryIndices())