import os
import numpy as np
from tqdm import tqdm
from lib.agents import Agent
from lib.mdps import BinaryMDP, SelfNonselfMDP

SIMULATOR_VERSION = "1.0.0"

class Simulator:
  
  def __init__(self, cache_dir=""):
    self._cache_dir = cache_dir

  def simulate(self, monitors=[], record_per=1000, max_epoch=100000, seed=0,
      n_key_states=30, dim_state=100, dim_action=20, mdp_type='BinaryMDP',
      n_hidden=5000, learning_rate=0.1, start_beta=1.0, last_beta=20):
    
    # Get the experiment ID
    exp_id = self._experiment_id(record_per, max_epoch, seed,
        n_key_states, dim_state, dim_action, mdp_type,
        n_hidden, learning_rate, start_beta, last_beta)
    exp_dir = os.path.join(self._cache_dir, exp_id)
    
    if self._cache_dir and os.path.exists(exp_dir):
      # If the exact same simulation was run before, get the result from cache
      epochs = np.load(os.path.join(exp_dir, "epochs.npy"))
      rewards = np.load(os.path.join(exp_dir, "rewards.npy"))
      agent_ns = np.load(os.path.join(exp_dir, "agent_ns.npy"))
      agent_w = np.load(os.path.join(exp_dir, "agent_w.npy"))
      agent_u = np.load(os.path.join(exp_dir, "agent_u.npy"))
      # key_states = np.load(os.path.join(exp_dir, "key_states.npy"))
      # key_actions = np.load(os.path.join(exp_dir, "key_actions.npy"))
      
    else:
      # If no cache were found, run new simulation
      (agent_w, agent_u, epochs, rewards, agent_ns) = self._simulate(
          record_per, max_epoch, seed,
          n_key_states, dim_state, dim_action, mdp_type,
          n_hidden, learning_rate, start_beta, last_beta)
      
      # Save results as a cache
      os.makedirs(exp_dir)
      np.save(os.path.join(exp_dir, "epochs.npy"), epochs)
      np.save(os.path.join(exp_dir, "rewards.npy"), rewards)
      np.save(os.path.join(exp_dir, "agent_ns.npy"), agent_ns)
      np.save(os.path.join(exp_dir, "agent_w.npy"), agent_w)
      np.save(os.path.join(exp_dir, "agent_u.npy"), agent_u)
    
    # Return results following the `monitors` parameter
    return_values = list()
    for monitor in monitors:
      if monitor == "epochs": return_values.append(epochs)
      elif monitor == "rewards": return_values.append(rewards)
      elif monitor == "agent_ns": return_values.append(agent_ns)
      elif monitor == "agent_n": return_values.append(agent_ns[-1])
      elif monitor == "agent_w": return_values.append(agent_w)
      elif monitor == "agent_u": return_values.append(agent_u)
      else: raise ValueError("Argument {} is invalid".format(monitor))
    return return_values

  def _simulate(self, record_per, max_epoch, seed,
      n_key_states, dim_state, dim_action, mdp_type,
      n_hidden, learning_rate, start_beta, last_beta):
    # Fix random seed
    np.random.seed(seed=seed)
    
    # Initialize MDP
    if mdp_type == 'BinaryMDP':
      mdp = BinaryMDP(n_key_states, dim_state, dim_action)
    elif mdp_type == 'SelfNonselfMDP':
      mdp = SelfNonselfMDP(n_key_states, dim_state, dim_action)
    else:
      raise ValueError("There is no mdp named {}".format(mdp_type))
    
    # Initialize Agent
    agent = Agent(n_hidden, dim_state, dim_action)
    
    # Prepare storage to save histories
    epochs = np.empty(max_epoch // record_per)
    rewards = np.empty(max_epoch // record_per)
    agent_ns = np.empty((max_epoch // record_per, n_hidden))
    
    # Simulation
    state = mdp.initial_state()
    for epoch in tqdm(range(max_epoch)):
      beta = start_beta + (last_beta - start_beta) * epoch / max_epoch
      action = agent.play(state, beta)
      reward = mdp.reward(action)
      agent.learn(state, action, reward, lr=learning_rate)
      state = mdp.next_state(action)
      if (epoch+1) % record_per == 0:
        record_idx = (epoch+1) // record_per - 1
        epochs[record_idx] = epoch+1
        rewards[record_idx] = reward / mdp.optimal_reward()
        agent_ns[record_idx] = agent.n
    
    return (agent.w, agent.u, epochs, rewards, agent_ns)
  
  def _experiment_id(self, record_per, max_epoch, seed,
      n_key_states, dim_state, dim_action, mdp_type,
      n_hidden, learning_rate, start_beta, last_beta):
    return '-'.join([
      "VER={}".format(SIMULATOR_VERSION),
      "RECORD_PER={}".format(record_per),
      "MAX_EPOCH={}".format(max_epoch),
      "SEED={}".format(seed),
      "N_KEY_STATES={}".format(n_key_states),
      "DIM_STATE={}".format(dim_state),
      "DIM_ACTION={}".format(dim_action),
      "MDP_TYPE={}".format(mdp_type),
      "N_HIDDEN={}".format(n_hidden),
      "LEARNING_RATE={}".format(learning_rate),
      "START_BETA={}".format(start_beta),
      "LAST_BETA={}".format(last_beta)
    ])
