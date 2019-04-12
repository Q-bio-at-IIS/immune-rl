# -*- coding: utf-8 -*-
import os
import numpy as np
from lib.utils import matching_score, samp

class BinaryMDP:
    
  def __init__(self, n_key_states, dim_state, dim_action):
    self.key_states = np.random.randint(
        0, 2, size=(n_key_states, dim_state))
    self.key_actions = np.random.randint(
        0, 2, size=(n_key_states, dim_action))
    self.cur_state_idx = np.random.randint(0, n_key_states)
    
    self.n_key_states = n_key_states
    self.dim_state = dim_state
    self.dim_action = dim_action
  
  def initial_state(self):
    self.cur_state_idx = np.random.randint(0, self.n_key_states)
    return self.key_states[self.cur_state_idx, :]
  
  def next_state(self, action):
    lin = np.dot(1 - self.key_actions, action)
    self.cur_state_idx = np.argmax(samp(lin / np.sum(lin)))
    return self.key_states[self.cur_state_idx, :]
  
  def optimal_action(self):
    return self.key_actions[self.cur_state_idx, :]
  
  def reward(self, action):
    return matching_score(self.key_actions[self.cur_state_idx, :], action)
  
  def optimal_reward(self):
    return self.dim_action
  
  def save(self, log_dir):
    np.save(os.path.join(log_dir, "mdp_key_states"), self.key_states)
    np.save(os.path.join(log_dir, "mdp_key_actions"), self.key_actions)
