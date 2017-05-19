#!/usr/bin/env python

"""
    awarp.py
"""

import numpy as np
from numba import jit

L = 1
T = 2
INF = int(1e10)

@jit(nopython=True)
def ub_cases(a, b, mode):
    if (a > 0) and (b > 0):
        return (a - b) ** 2
    elif (a > 0) and (b < 0):
        if mode == L:
            return a ** 2
        else:
            return -b * (a ** 2)
    elif (a < 0) and (b > 0):
        if mode  == T:
            return b ** 2
        else:
            return -a * (b ** 2)
    else:
        return 0

@jit(nopython=True)
def awarp_(D, s, t):
    for i in range(s.shape[0]):
        for j in range(t.shape[0]):
            
            if (i > 0) and (j > 0):
                a_d = D[i,j] + ub_cases(s[i], t[j], 0)
            else:
                a_d = D[i,j] + (s[i] - t[j]) ** 2
            
            a_t = D[i+1,j] + ub_cases(s[i], t[j], T)
            a_l = D[i,j+1] + ub_cases(s[i], t[j], L)
            
            D[i+1, j+1] = min(a_d, a_t, a_l)


def awarp(s, t, return_matrix=False):
    D = np.zeros((s.shape[0] + 1, t.shape[0] + 1)).astype('int')
    D[:,0] = int(INF)
    D[0,:] = int(INF)
    D[0,0] = 0
    awarp_(D, s, t)
    if return_matrix:
        return D[1:,1:]
    else:
        return D[-1,-1]
    
