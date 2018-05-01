#!/usr/bin/env python

"""
    awarp.py
"""

import numpy as np
from numba import jit

L = 1
T = 2
INF = int(2 ** 64 - 1)

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
        if mode == T:
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
    return D[1:,1:] if return_matrix else D[-1,-1]

# --

@jit(nopython=True)
def compute_t(x):
    xt = np.zeros(x.shape[0] + 1, dtype=numba.int32)
    
    idx = 0
    for i in range(s.shape[0]):
        if x[i] > 0:
            idx += 1
        else:
            idx += abs(x[i])
        
        xt[i] = idx
    
    xt[-1] = idx + 1
    return xt

@jit(nopython=True)
def constrained_awarp_(D, s, t, w):
    st = compute_t(s)
    tt = compute_t(t)
    
    for i in range(s.shape[0]):
        for j in range(t.shape[0]):
            
            gap = abs(st[i] - tt[j])
            if (
                (gap > w) and 
                ((j > 0) and (tt[j - 1] - st[i] > w)) and 
                ((i > 0) and (st[i - 1] - tt[j] > w))
            ):
                D[i+1, j+1] = INF
            else:
                
                if (i > 0) and (j > 0):
                    a_d = D[i,j] + ub_cases(s[i], t[j], 0)
                else:
                    a_d = D[i,j] + (s[i] - t[j]) ** 2
                
                a_t = D[i+1,j] + ub_cases(s[i], t[j], T)
                a_l = D[i,j+1] + ub_cases(s[i], t[j], L)
                
                D[i+1, j+1] = min(a_d, a_t, a_l)


def constrained_awarp(s, t, w, return_matrix=False):
    D = np.zeros((s.shape[0] + 1, t.shape[0] + 1)).astype('int')
    D[:,0] = INF
    D[0,:] = INF
    D[0,0] = 0
    constrained_awarp_(D, s, t, w)
    return D[1:,1:] if return_matrix else D[-1,-1]


s = np.random.choice((-3, 1), 100)
t = np.random.choice((-3, 1), 100)
awarp(s, t)
# constrained_awarp(s, t, w=1000, return_matrix=True)


def to_dense(x):
    out = []
    for i in range(len(x)):
        if x[i] > 0:
            out.append(1)
        else:
            out += [0] * abs(x[i])
    
    return out

ds = to_dense(s)
dt = to_dense(t)

fastdtw.dtw(ds, dt)[0]
