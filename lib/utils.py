# -*- coding: utf-8 -*-
import numpy as np

def samp(p):
  return np.random.binomial(1, p)

def relu(x):
  return np.maximum(0, x)

def step(x):
  return x > 0

def matching_score(a, b):
  return np.sum(np.logical_not(np.logical_xor(a, b)))
