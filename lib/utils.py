import numpy as np
from collections import Counter

def samp(p):
  return np.random.binomial(1, p)

def relu(x):
  return np.maximum(0, x)

def step(x):
  return x > 0

def matching_score(a, b):
  return np.sum(np.logical_not(np.logical_xor(a, b)))

def multinomial(dist, total_num):
  normed = dist / np.sum(dist)
  samples = np.random.choice(len(normed), total_num, p=normed)
  counts = Counter(samples)
  return np.array(list(counts.values()))
