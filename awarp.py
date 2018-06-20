#!/usr/bin/env python

"""
    awarp.py
"""

import numpy as np
import numba

# --
# Helpers

def to_dense(x):
    out = []
    for i in range(len(x)):
        if x[i] > 0:
            out.append(1)
        else:
            out += [0] * abs(x[i])
    
    return np.array(out)

def run_encode(x, v=None):
    out = np.ones(2 * len(x) - 1, dtype=np.int)
    out[1::2] = -(np.diff(x) - 1)
    if v is not None:
        out[0::2] = v
    return out


# --
# Unconstrained

L = 1
T = 2
INF = int(1e10)

ZERO_PENALTY = 0
@numba.jit(nopython=True)
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
        return ZERO_PENALTY * abs(a - b)


@numba.jit(nopython=True)
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


# --
# Constrained

@numba.jit(nopython=True)
def compute_t(x):
    xt = np.zeros(x.shape[0] + 1, dtype=numba.int32)
    
    idx = 0
    for i in range(x.shape[0]):
        if x[i] > 0:
            idx += 1
        else:
            idx += abs(x[i])
        
        xt[i] = idx
    
    xt[-1] = idx + 1
    return xt


@numba.jit(nopython=True)
def constrained_awarp_(D, s, t, w):
    st = compute_t(s)
    tt = compute_t(t)
    
    for i in range(s.shape[0]):
        for j in range(t.shape[0]):
            
            too_far = abs(st[i] - tt[j]) > w
            jcond   = (j > 0) and (tt[j - 1] - st[i] > w)
            icond   = (i > 0) and (st[i - 1] - tt[j] > w)
            if too_far and (jcond or icond):
                D[i+1, j+1] = INF
            else:
                
                if (i > 0) and (j > 0):
                    a_d = D[i,j] + ub_cases(s[i], t[j], 0)
                else:
                    a_d = D[i,j] + (s[i] - t[j]) ** 2
                
                a_t = D[i+1,j] + ub_cases(s[i], t[j], T)
                a_l = D[i,j+1] + ub_cases(s[i], t[j], L)
                
                D[i+1, j+1] = min(a_d, a_t, a_l)

# --
# Wrapper

def awarp(s, t, w=None, return_matrix=False, preencode=True):
    if preencode:
        s = run_encode(s)
        t = run_encode(t)
    
    # assert s[0] == 1, "s[0] != 1"
    # assert t[0] == 1, "t[0] != 1"
    
    D = np.zeros((s.shape[0] + 1, t.shape[0] + 1)).astype(int)
    D[:,0] = INF
    D[0,:] = INF
    D[0,0] = 0
    
    if w is None:
        awarp_(D, s, t)
    else:
        constrained_awarp_(D, s, t, w)
    
    if return_matrix:
        D = D[1:,1:].astype('float')
        D[D == INF] = np.inf
        return D
    else:
        d = D[-1,-1]
        return d if d != INF else np.inf
