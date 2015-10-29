# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:00:45 2015

@author: Achille
"""

import numpy as np
from scipy import linalg
from kernelfun import rbf_dot



def kernel_derivative(X, Y, K, sigma_x, sigma_y, eps, verbose=True):
    """
    computes initial estimate of SDR matrix by gradient descent
    
    """
    
    n, d = X.shape
    #gram matrix of X
    Kx  = rbf_dot(X, X, sigma_x)
    Kxi = linalg.inv(Kx + n*eps*np.eye(n))
    
    #Gram matrix of Y
    Ky = rbf_dot(Y, Y, sigma_y)
    
    #Derivative of Kx(xi, x) w.r.t. x
    Dx = np.reshape(np.tile(X, n, 1), (n,n,d))
    Xij = Dx - Dx.transpose((0, 2, 1))
    Xij = Xij/(sigma_x**2)
    H = Xij*np.reshape(np.tile(Kx, 1, d), (n,n,d))
    
    #sum_i H(X_i)'*Kxi*Ky*Kxi*H(X_i)
    
    Fmat = np.dot(Kxi, np.dot(Ky, Kxi))
    Hd = H.reshape((n, n*d))
    HH = np.reshape(np.dot(Hd.T, Hd), (n,d,n,d))
    HHd = np.reshape(np.transpose(HH, (1,3,2,4)), (n**2,d,d)) 
    Fd = np.reshape(np.reshape(Fmat, (n**2,1)), (n**2,d,d))
    
    R = np.reshape(np.sum(HHd*Fd, axis=0), (d,d))
    L, V = linalg.eigh(R)
    B = V[:,::-1][:,:K]
    L = L[::-1]
    tr = np.sum(L[:K])
    
    return B, tr