#!/usr/bin/env python

"""
    test.py
"""

from __future__ import print_function

import sys
import fastdtw
import numpy as np
from tqdm import tqdm
from awarp import awarp, to_dense

for _ in tqdm(range(100)):
    s = np.sort(np.random.choice(1000, 100))
    t = np.sort(np.random.choice(1000, 100))
    
    new = awarp(s, t, return_matrix=False)
    
    ds = to_dense(s)
    dt = to_dense(t)
    base = fastdtw.dtw(ds, dt, dist=None)[0]
    
    assert new == base


s = np.array([1, -1, 1, -1, 1])
t = np.array([1, -1, 1, -5, 1])
awarp(s, t, preencode=False, return_matrix=True)
awarp(s, t, w=4, preencode=False, return_matrix=True)

