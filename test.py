#!/usr/bin/env python

"""
    test.py
"""

import numpy as np
from awarp import awarp

s = np.random.choice((-3, 1), 100)
t = np.random.choice((-3, 1), 100)
awarp(s, t, return_matrix=True)
