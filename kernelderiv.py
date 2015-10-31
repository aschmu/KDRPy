# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:00:45 2015

@author: Achille
"""

import numpy as np
from scipy import linalg, optimize
from kernelfun import rbf_dot



def kernel_derivative(X, Y, K, sigma_x, sigma_y, eps):
    """
    computes initial estimate of SDR matrix by gradient descent
    
    arguments :
    X       -- nxd array of n samples, d features
    Y       -- nxp array of class labels
    K       -- target dimension of SDR subspace
    sigma_x -- scale factor for the Gaussian kernel associated to X
    sigma_y -- scale factor for the Gaussian kernel associated to Y
    eps     -- regularization factor for matrix inversion
    
    returns :
    B       -- initial SDR matrix estimate after gradient descent
    tr      -- corresponding trace value (trace=objective function)        
    
    """
    
    n, d = X.shape
    #gram matrix of X
    Kx  = rbf_dot(X, X, sigma_x)
    Kxi = linalg.inv(Kx + n*eps*np.eye(n))
    
    #Gram matrix of Y
    Ky = rbf_dot(Y, Y, sigma_y)
    
    #Derivative of Kx(xi, x) w.r.t. x
    Dx = np.reshape(np.tile(X, (n, 1)), (n,n,d))
    Xij = Dx - Dx.transpose((1, 0, 2))
    Xij = Xij/(sigma_x**2)
    H = H = Xij*np.reshape(np.tile(Kx,( 1, d)), (n,n,d)) #Xij*np.tile(Kx,(1,1,d)) #
    
    #sum_i H(X_i)'*Kxi*Ky*Kxi*H(X_i)
    
    Fmat = np.dot(Kxi, np.dot(Ky, Kxi))
    Hd = H.reshape((n, n*d))
    HH = np.reshape(np.dot(Hd.T, Hd), (n,d,n,d))
    HHd = np.reshape(np.transpose(HH, (0,2,1,3)), (n**2,d,d)) 
    Fd = np.tile(np.reshape(Fmat, (n**2,1,1)), (1,d,d))
    
    R = np.reshape(np.sum(HHd*Fd, axis=0), (d,d))
    L, V = linalg.eigh(R)
    B = V[:,::-1][:,:K]
    L = L[::-1]
    tr = np.sum(L[:K])    
    
    return B, tr
    
    
    
def KDR_linesearch(X, Ky, sz2, B, dB, eta, eps, ls_maxiter):
    """
    line search step for the minimization of Tr[ Ky(Kz(B)+eps*I)^{-1} ] 
    where Ky and Kz(B) are the centered Gram matrices computed using Gaussian kernels 

    arguments:
    X          -- nxd array
    Ky         -- centered Y Gram matrix
    sz2        -- (annealed) Gaussian kernel scale factor for Kz Gram matrix
    B          -- current iteration SDR matrix
    dB         -- SDR matrix derivative
    eta        -- upper bound of the minimization region [0, eta]
    eps        -- regularization term 
    ls_maxiter -- max number of iterations during line search step size selection
    
    returns:
    Bn         -- B - s*dB where s is the stepsize parameter
    tr         -- trace value for the annealed scale factor sz2
    
    """

    n = X.shape[0]
    Q = np.eye(n) - np.ones((n,n))/n
    
    def kdrobjfun1D(s):
        tmpB = B - s*dB
        tmpB = linalg.svd(tmpB, full_matrices=False)[0]
        Z    = np.dot(X, tmpB)
        Kz   = rbf_dot(Z, Z, np.sqrt(sz2))
        Kz   = np.dot(np.dot(Q,Kz), Q)
        Kz   = (Kz + Kz.T)/2 
        
        t = np.sum(Ky*linalg.inv(Kz + n*eps*np.eye(n)))
        
        return t
    #try adding options to minimize nb of optim steps    
    res = optimize.minimize_scalar(kdrobjfun1D, bounds=(0, eta),
                                   method='bounded', 
                                   options={'maxiter':ls_maxiter, 'disp':False})
    s   = res.x   
    tr  = res.fun
    Bn  = B - s*dB
    
    return Bn, tr
    
    
        