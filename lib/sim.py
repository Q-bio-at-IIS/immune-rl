# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from lib.agents import Agent
from lib.mdps import BinaryMDP

SIMULATOR_VERSION = "1.0.0"

class Simulator:
  
  def __init__(self, cache_dir=""):
    self._cache_dir = cache_dir

  def simulate(self, monitors=[], record_per=1000, max_epoch=100000, seed=0,
      n_key_states=30, dim_state=100, dim_action=20,
      n_hidden=5000, learning_rate=0.1, start_beta=1.0, last_beta=20):
    
    # このシミュレーションのIDを取得する
    exp_id = self._experiment_id(record_per, max_epoch, seed,
        n_key_states, dim_state, dim_action,
        n_hidden, learning_rate, start_beta, last_beta)
    exp_dir = os.path.join(self._cache_dir, exp_id)
    
    if self._cache_dir and os.path.exists(exp_dir):
      # すでに同じシミュレーションを行なっていた場合にはそのデータをロードする
      epochs = np.load(os.path.join(exp_dir, "epochs.npy"))
      rewards = np.load(os.path.join(exp_dir, "rewards.npy"))
      agent_ns = np.load(os.path.join(exp_dir, "agent_ns.npy"))
      agent_w = np.load(os.path.join(exp_dir, "agent_w.npy"))
      agent_u = np.load(os.path.join(exp_dir, "agent_u.npy"))
      key_states = np.load(os.path.join(exp_dir, "key_states.npy"))
      key_actions = np.load(os.path.join(exp_dir, "key_actions.npy"))
      
    else:
      # まだ行なっていない場合には新たにシミュレーションを行う
      (agent_w, agent_u, key_states, key_actions,
        epochs, rewards, agent_ns) = self._simulate(
          record_per, max_epoch, seed,
          n_key_states, dim_state, dim_action,
          n_hidden, learning_rate, start_beta, last_beta)
      
      # キャッシュとして保存しておく
      os.makedirs(exp_dir)
      np.save(os.path.join(exp_dir, "epochs.npy"), epochs)
      np.save(os.path.join(exp_dir, "rewards.npy"), rewards)
      np.save(os.path.join(exp_dir, "agent_ns.npy"), agent_ns)
      np.save(os.path.join(exp_dir, "agent_w.npy"), agent_w)
      np.save(os.path.join(exp_dir, "agent_u.npy"), agent_u)
      np.save(os.path.join(exp_dir, "key_states.npy"), key_states )
      np.save(os.path.join(exp_dir, "key_actions.npy"), key_actions)
    
    # monitorsにしたがって返す値を決める
    return_values = list()
    for monitor in monitors:
      if monitor == "epochs": return_values.append(epochs)
      elif monitor == "rewards": return_values.append(rewards)
      elif monitor == "agent_ns": return_values.append(agent_ns)
      elif monitor == "agent_n": return_values.append(agent_ns[-1])
      elif monitor == "agent_w": return_values.append(agent_w)
      elif monitor == "agent_u": return_values.append(agent_u)
      elif monitor == "key_states": return_values.append(key_states)
      elif monitor == "key_actions": return_values.append(key_actions)
      else: raise ValueError("Argument {} is invalid".format(monitor))
    return return_values

  def _simulate(self, record_per, max_epoch, seed,
      n_key_states, dim_state, dim_action,
      n_hidden, learning_rate, start_beta, last_beta):
    # ランダムシードを固定
    np.random.seed(seed=seed)
    
    # MDPとAgentを初期化
    mdp = BinaryMDP(n_key_states, dim_state, dim_action)
    agent = Agent(n_hidden, dim_state, dim_action)
    
    # 学習中の経過を記録するためのコンテナ用意
    epochs = np.empty(max_epoch // record_per)
    rewards = np.empty(max_epoch // record_per)
    agent_ns = np.empty((max_epoch // record_per, n_hidden))
    
    # 学習
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
    
    return (agent.w, agent.u, mdp.key_states, mdp.key_actions,
        epochs, rewards, agent_ns)
  
  def _experiment_id(self, record_per, max_epoch, seed,
      n_key_states, dim_state, dim_action,
      n_hidden, learning_rate, start_beta, last_beta):
    return '-'.join([
      "VER={}".format(SIMULATOR_VERSION),
      "RECORD_PER={}".format(record_per),
      "MAX_EPOCH={}".format(max_epoch),
      "SEED={}".format(seed),
      "N_KEY_STATES={}".format(n_key_states),
      "DIM_STATE={}".format(dim_state),
      "DIM_ACTION={}".format(dim_action),
      "N_HIDDEN={}".format(n_hidden),
      "LEARNING_RATE={}".format(learning_rate),
      "START_BETA={}".format(start_beta),
      "LAST_BETA={}".format(last_beta)
    ])
