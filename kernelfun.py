# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:49:49 2015

utilities for standard kernel operations 

@author: Achille
"""

import numpy as np
from scipy.spatial.distance import pdist


def rbf_dot(X, Y, sigma):
    """
    computes Gram matrix using a Gaussian kernel
    
    arguments:
    X     --  nxd array
    Y     --  nxd array
    sigma --  kernel width
    """
    
    if X.shape[1] != Y.shape[1] :
        raise ValueError('X and Y must have same dimensions')
    
    nx, d = X.shape
    ny    = Y.shape[0]
    
    G = (X**2).sum(axis=1) 
    H = (Y**2).sum(axis=1)
    
    Q = np.tile(G[:,np.newaxis],(1,ny))
    R = np.tile(H, (nx,1))
    
    K = Q + R -2*np.dot(X, Y.T)
    
    return K
 

def estim_sigmakernel_median(X, nb_samples_max):
    """
    provides an estimate of the Gaussian kernel width parameter based on
    the median of the between-samples distance
    
    arguments:
    X               -- n x d array
    nb_samples_max  -- number of maximum samples used for the computation 
                        of the median
    """
    
    m = X.shape[0]
    
    if m > nb_samples_max:
        isub   = np.random.choice(m, nb_samples_max, replace=False)
        dist_X = pdist(X[isub,:])
    else:
        dist_X = pdist(X)
        
    sigma  = np.median(dist_X)
    
    return sigma
    
    