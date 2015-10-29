# -*- coding: utf-8 -*-
"""

Kernel Dimension Reduction (KDR) routines

see "Kernel dimension reduction in regression" by K.Fukumizu, Bach F.
 and M.I. Jordan, Annals of Statistics, 2009

@author: Achille
"""

import numpy as np
from scipy import linalg
from kernelfun import rbf_dot, estim_sigmakernel_median
from kernelderiv import kernel_derivative

def kdr_optim(X, Y, K, max_loop, sigma_x, sigma_y, eps,
              eta, anl, verbose = True, tol=1e-9, 
              init_deriv = False):
    
        n, d  = X.shape
        
        if n != Y.shape[0]:
            raise ValueError, 'X and Y have incompatible dimensions'
         
        assert K<=d, 'dimension K must be lower than d !' 
        if init_deriv:
            print 'Initialization by derivative method...\n'
            B, t = kernel_derivative(X, Y, K, np.sqrt(anl)*sigma_x,
                                     sigma_y, eps)
        else:            
            B = np.random.randn(d, K)
        
        B = linalg.svd(B)[0]
                
        unit = np.ones((n,n))
        Q = np.eye(n) - unit/n
        
        """Gram matrix of Y"""
        Gy  = rbf_dot(Y, Y, sigma_y) #check this 
        Kyo = np.dot(np.dot(Q, Gy), Q)
        Kyo  = (Kyo + Kyo.T)/2
        
        """objective function initial value """
        Z = np.dot(X, B)
#        nZ = Z/np.sqrt(2)/sigma_x1
        Gz = rbf_dot(Z, Z, sigma_x)
        Kz = np.dot(np.dot(Q, Gz), Q)
        Kz = (Kz + Kz.T)/2
        
        mz = linalg.inv(Kz + eps*n*np.diag(n))
        tr = np.sum(Kyo*mz)
        
        if verbose:
            print '[0] trace = ', tr
        
        ssz2 = 2*sigma_x**2
        ssy2 = 2*sigma_y**2
        #careful h from 0 to maxloop-1, implement accordingly
        for h in xrange(max_loop): 
            sz2 = ssz2+(anl-1)*ssz2*(max_loop-h+1)/max_loop
            sy2 = ssy2+(anl-1)*ssy2*(max_loop-h+1)/max_loop
            
            Z  = np.dot(X, B)
            Kzw = rbf_dot(Z, Z, np.sqrt(sz2))
            Kzi = linalg.inv(np.dot(np.dot(Q, Kzw), Q) + eps*n*np.eye(n)) #forget about Kz don't really need it afterwards
            
            Ky = rbf_dot(Y, Y, np.sqrt(sy2))
            Ky = np.dot(np.dot(Q, Ky), Q)
            #Kyzi = np.dot(Ky, Kzi) inutile de le déclarer
            
            dB = np.zeros((d,K))
            KziKyzi = np.dot(Kzi, np.dot(Ky, Kzi))
            
            for a in xrange(d):
                Xa = np.tile(X[:,1][:,np.newaxis], 1, n)
                XX = Xa - Xa.T
                for b in xrange(K):
                    Zb = np.tile(Z[:,b][:,np.newaxis], 1, n)
                    tt = XX*(Zb - Zb.T)*Kzw
                    dKB = np.dot(Q, np.dot(tt, Q))
                    dB[a, b] = np.sum(KziKyzi*dKB.T)
            
            nm = linalg.norm(dB, 2)
            if nm < tol:
                break
            B, tr = KDR_linesearch(X, Ky, sz2, B, dB/nm, eta, eps)
            B = linalg.svd(B)[0]
           
            """ compute trace with unannealed parameter"""
            if verbose:
                Z = np.dot(X, B)
                Kz = rbf_dot(Z, Z, sigma_x)
                Kz = np.dot(np.dot(Q, Kz), Q)
                Kz = (Kz + Kz.T)/2
                tr = np.sum(Kyo*mz)
                print '[%d] trace = %.6f \n'  % (h,tr) 
        
        return B
           