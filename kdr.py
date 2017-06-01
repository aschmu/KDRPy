# -*- coding: utf-8 -*-
"""

Kernel Dimension Reduction (KDR) 
----------------------------------

Performs sufficient dimension reduction for the regression/classification 
of Y (response) on X where X is a vector of predictors in R^d, and d>>1 possibly.
The goal is to find low rank matrix B such that B^TX is a sufficient predictor of Y
i.e. Y|X ~ Y|B^TX. 
The method proceeds by minimizing Tr[ Ky(Kz(B)+eps*I_n)^{-1} ] w.r.t B
Ky and Kz are (characteristic) kernels associated to Y and Z=B^TX, eps is a
regularization parameter and n is the sample size.

This implementation assumes Gaussian kernels and is based on K. Fukumizu's matlab code.

References:
-----------

"Kernel dimension reduction in regression" by K.Fukumizu, Bach F.
and M.I. Jordan, Annals of Statistics, 2009


@author: Achille
"""
#from __future__ import print_function
import numpy as np
from scipy import linalg
from kernelfun import rbf_dot, estim_sigmakernel_median, center_matrix
from kernelderiv import kernel_derivative, KDR_linesearch
from sklearn import preprocessing
from matplotlib import pyplot as plt
from builtins import range

def kdr_optim(X, Y, K, max_loop, sigma_x, sigma_y, eps,
              eta, anl, verbose = True, tol=1e-9, 
              init_deriv = False, ls_maxiter=30):
    """                                             
    arguments :
    X           -- nxd array of n samples, d features
    Y           -- nxp array of class labels
    K           -- target dimension of SDR subspace
    max_loop    -- maximum number of iterations    
    sigma_x     -- scale factor for the Gaussian kernel associated to X (float)
    sigma_y     -- scale factor for the Gaussian kernel associated to Y (float)
    eps         -- regularization factor for matrix inversion (float)
    eta         -- upper bound for linesearch step parameter (float)
    anl         -- maximum annealing parameter (int/float)
    verbose     -- print objective function value at each iteration ? (bool)
    tol         -- stopping criterion for gradient descent, ie 
                   optim stops when ||dB||_s < tol (float) where ||.||_s is the
                   spectral norm
    init_deriv  -- use initial estimate of B through gradient descent ? (bool)
    ls_maxiter  -- max number of iterations during line search step size selection (int)
    
    returns :
    B           -- SDR matrix estimate 
    
    """
    n, d  = X.shape
            
    if n != Y.shape[0]:
        raise(ValueError('X and Y have incompatible dimensions'))
     
    assert K<=d, 'dimension K must be lower than d !'
    assert sigma_x > 0 and sigma_y > 0, 'scale parameters must be positive!'
    assert tol > 0, 'tolerance factor must be >0'
    
    if init_deriv:
        print('Initialization by derivative method...\n')
        B, t = kernel_derivative(X, Y, K, np.sqrt(anl)*sigma_x,
                                 sigma_y, eps)
    else:            
        B = np.random.randn(d, K)
    
    B = linalg.svd(B, full_matrices=False)[0]
                
    """Gram matrix of Y"""
    Gy  = rbf_dot(Y, Y, sigma_y) 
    Kyo = center_matrix(Gy) 
    Kyo  = (Kyo + Kyo.T)/2
    
    """objective function initial value """
    Z = np.dot(X, B)
    Gz = rbf_dot(Z, Z, sigma_x)
    Kz = center_matrix(Gz) 
    Kz = (Kz + Kz.T)/2
    
    mz = linalg.inv(Kz + eps*n*np.eye(n))
    tr = np.sum(Kyo*mz)
    
    if verbose:
        print('[0]trace = {}'.format(tr))
    
    ssz2 = 2*sigma_x**2
    ssy2 = 2*sigma_y**2
    #careful h from 0 to maxloop-1, implement accordingly
    for h in range(max_loop): 
        sz2 = ssz2+(anl-1)*ssz2*(max_loop-h-1)/max_loop
        sy2 = ssy2+(anl-1)*ssy2*(max_loop-h-1)/max_loop
        
        Z  = np.dot(X, B)
        Kzw = rbf_dot(Z, Z, np.sqrt(sz2))
        Kz  = center_matrix(Kzw) 
        Kzi = linalg.inv(Kz + eps*n*np.eye(n)) #
        
        Ky = rbf_dot(Y, Y, np.sqrt(sy2))
        Ky = center_matrix(Ky) 
        Ky = (Ky + Ky.T)/2
         
        
        dB = np.zeros((d,K))
        KziKyzi = np.dot(Kzi, np.dot(Ky, Kzi))
        
        for a in range(d):
            Xa = np.tile(X[:,a][:,np.newaxis], (1, n))
            XX = Xa - Xa.T
            for b in range(K):
                Zb = np.tile(Z[:,b][:,np.newaxis], (1, n))
                tt = XX*(Zb - Zb.T)*Kzw
                dKB = center_matrix(tt) 
                dB[a, b] = np.sum(KziKyzi*dKB.T) #np.trace(np.dot(Kzi.dot(Kyzi),dKB))  #
        
        nm = linalg.norm(dB, 2)
        if nm < tol:
            break
        B, tr = KDR_linesearch(X, Ky, sz2, B, dB/nm, eta, eps,
                               ls_maxiter=ls_maxiter)
        B = linalg.svd(B, full_matrices=False)[0]
       
        """ compute trace with unannealed parameter"""
        if verbose:
            Z = np.dot(X, B)
            Kz = rbf_dot(Z, Z, sigma_x)
            Kz = center_matrix(Kz) #np.dot(np.dot(Q, Kz), Q)
            Kz = (Kz + Kz.T)/2
            mz = linalg.inv(Kz + n*eps*np.eye(n))
            tr = np.sum(Kyo*mz)
            print('[%d]trace = %.6f'  % (h+1,tr) )
    
    return B
        
        
if __name__ == "__main__":
    
    max_iter = 50
    epsilon = 1e-4
    eta_linesearch = 10.0
    verbose = True
    annealing = 4
    r = 2 #dimension of SDR subspace
    
    
    print('KDR demo using wine data from UCI Repository')
    
    data = np.genfromtxt(fname='./data/wine_data.csv', delimiter=",")    
    y = data[:, 0][:,np.newaxis]
    X = data[:, 1:]
    d = X.shape[1]
    N = X.shape[0]
    
    print(d, 'features,', N, 'samples and', 3, 'classes')

    #standardize data
    std_scaler = preprocessing.StandardScaler().fit(X)    
    Xscaled = std_scaler.transform(X)
    
    #estimate Gaussian scale parameter
    sigma_X = 0.5*estim_sigmakernel_median(Xscaled)
    sigma_y = estim_sigmakernel_median(y)
    
    B = kdr_optim(X=Xscaled, Y=y, K=r, max_loop=max_iter, sigma_x=sigma_X*np.sqrt(np.float(r)/d), 
                  sigma_y=sigma_y, eps=epsilon, eta=eta_linesearch, 
                  anl=annealing, verbose=verbose, init_deriv=False)
           
    Z = np.dot(Xscaled, B)
    plt.scatter(Z[np.ravel(y)==1,0], Z[np.ravel(y)==1,1], color="blue", label='class 1')
    plt.scatter(Z[np.ravel(y)==2,0], Z[np.ravel(y)==2,1], color="red", label='class 2')
    plt.scatter(Z[np.ravel(y)==3,0], Z[np.ravel(y)==3,1], color="green", label='class 3')
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.title('Projections of wine data on first 2 SDR directions')
    plt.legend()
    
    