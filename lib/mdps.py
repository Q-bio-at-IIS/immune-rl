import os
import numpy as np
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from lib.utils import matching_score, samp

# ----
# FixedOptimalRewardMDP
# Interface of MDPs whose maximum reward is fixed for all states
# ----
class FixedOptimalRewardMDP(metaclass=ABCMeta):
  @abstractmethod
  def initial_state(self): raise NotImplementedError()
  @abstractmethod
  def next_state(self, action): raise NotImplementedError()
  @abstractmethod
  def reward(self, action): raise NotImplementedError()
  @abstractmethod
  def optimal_reward(self): raise NotImplementedError()

# ----
# BinaryMDP
# A MDP with adversarial transition dynamics
# ----
class BinaryMDP(FixedOptimalRewardMDP):
    
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


# ----
# SelfNonselfMDP
# A MDP with multiple pathogenic states and one healthy state
# ----

# is_pathogenic: indicates whether the current state is pathogenic or not
# pathogen_idx: indicates which pathogen the host is infected
State = namedtuple('State', ['is_pathogenic', 'pathogen_idx'])
class SelfNonselfMDP(FixedOptimalRewardMDP):

  def __init__(self, n_pathogen, dim_state, dim_action, infection_rate=0.5):
    self.nonpathogenic_state = np.random.randint(0, 2, size=dim_state)
    self.pathogenic_states = np.random.randint(
        0, 2, size=(n_pathogen, dim_state))
    self.nonpathogenic_action = np.random.randint(0, 2, size=dim_action)
    self.pathogenic_actions = np.random.randint(
        0, 2, size=(n_pathogen, dim_action))
    self.cur_state = State(False, np.nan)
    
    self.n_pathogen = n_pathogen
    self.infection_rate = infection_rate
    self.dim_state = dim_state
    self.dim_action = dim_action
  
  def initial_state(self):
    if self.cur_state.is_pathogenic:
      return self.pathogenic_states[self.cur_state.pathogen_idx, :]
    else:
      return self.nonpathogenic_state
  
  def next_state(self, action):
    if self.cur_state.is_pathogenic:
      # If already infected by a pathogen,
      
      # action determines the probability of its elimination
      actionに応じて病原体を排除できる確率が定まり
      effective_action = self.pathogenic_actions[self.cur_state.pathogen_idx, :]
      ham_dist = matching_score(effective_action, action)
      eliminate_prob = ham_dist / self.dim_action
      
      if np.random.rand() < eliminate_prob:
        # If effective, the state goes back to heathly state
        self.cur_state = State(False, np.nan)
        return self.nonpathogenic_state
        
      else:
        # If not, the state stays the same
        return self.pathogenic_states[self.cur_state.pathogen_idx, :]
      
    else:
      # If not infected by a pathogen,
      
      if np.random.rand() < self.infection_rate:
        # infected by a pathogen in a fixed probability
        self.cur_state = State(True, np.random.choice(self.n_pathogen))
        return self.pathogenic_states[self.cur_state.pathogen_idx, :]
      
      else:
        self.cur_state = State(False, np.nan)
        return self.nonpathogenic_state
  
  def optimal_action(self):
    if self.cur_state.is_pathogenic:
      return self.pathogenic_actions[self.cur_state.pathogen_idx, :]
    else:
      return self.nonpathogenic_action
  
  def reward(self, action):
    if self.cur_state.is_pathogenic:
      effective_action = self.pathogenic_actions[self.cur_state.pathogen_idx, :]
    else:
      effective_action = self.nonpathogenic_action
    return matching_score(effective_action, action)
  
  def optimal_reward(self):
    return self.dim_action
