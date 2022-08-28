import os
import numpy as np
from scipy.special import expit

from lib.utils import samp

class Agent:
    
  def __init__(self, n_hidden, dim_state, dim_action):
    self.n = np.ones((n_hidden, ))
    
    scale = np.sqrt(2 / dim_state)
    self.w = np.random.normal(
        scale=scale, size=(n_hidden, dim_state))
    scale = np.sqrt(2 / n_hidden)
    self.u = np.random.normal(
        scale=scale, size=(dim_action, n_hidden))
    
    self.n_hidden = n_hidden
    self.dim_state = dim_state
    self.dim_action = dim_action
  
  def play(self, state, beta):
    self.h = expit(np.dot(self.w, state))
    self.nh = self.n * self.h
    action = samp(expit(beta * np.dot(self.u, self.nh)))
    return action
  
  def learn(self, state, action, reward, lr=0.01):
    q = np.dot(np.dot(self.u, self.nh), action)
    lam = (reward - q) * self.h * np.dot(self.u.T, action)
    self.n += lr * self.n * lam
    self.n = np.maximum(self.n, 0.0) # To ensure n >= 0
  
  def save(self, log_dir):
    np.save(os.path.join(log_dir, "agent_n"), self.n)
    np.save(os.path.join(log_dir, "agent_w"), self.w)
    np.save(os.path.join(log_dir, "agent_u"), self.u)
