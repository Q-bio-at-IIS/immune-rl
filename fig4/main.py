#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

sys.path.append("..")
from lib.sim import Simulator

def extract_data(exp_id):
  input_path = os.path.join(EXP_DATA_DIR, exp_id, EXP_DATA_FILE)
  data = pd.read_csv(input_path, delimiter=EXP_DELIMITER)
  population = data.groupby('aaSeqCDR3')['cloneCount'].sum().values
  return population.astype(np.int)

def plot_experiments(axis, population):
  print('-- Plot experiment')
  axis.loglog(sorted(population, reverse=True), color='#4C72B0')

def multinomial(agent_n, total_num):
  normed = agent_n / np.sum(agent_n)
  samples = np.random.choice(len(normed), total_num, p=normed)
  counts = Counter(samples)
  return np.array(list(counts.values()))

def plot_simulation(axis, agent_ns, total_population):
  print('-- Plot simulation')
  for agent_n in agent_ns:
    sampled = multinomial(agent_n, total_population)
    axis.loglog(sorted(sampled, reverse=True), color='#d0d0d0', linewidth=0.2)

def format_figure(axis, ylabel=None, labelbottom=False):
  axis.spines['top'].set_visible(False)
  axis.spines['right'].set_visible(False)
  axis.spines['left'].set_visible(False)
  axis.set_xlim([1, 4000])
  axis.set_ylim([1, 30])
  labelleft = True if ylabel else False
  axis.tick_params(axis='y', which='both', top=False, left=False, right=False,
      labelbottom=labelbottom, labelleft=labelleft)
  axis.grid(True, "minor", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
  axis.grid(True, "major", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
  axis.set_ylabel(ylabel)

def main():
  plot_rows = 2 # BLN_EXPS and PLN_EXPS
  plot_cols = max(len(BLN_EXPS), len(PLN_EXPS))
  fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, sharey=True)
  
  print('Start simulation')
  simulator = Simulator(CACHE_DIR)
  agent_ns = np.empty((N_SIM, N_HIDDEN))
  for idx in range(N_SIM):
    print('-- Start', idx, 'th simulation')
    agent_ns[idx, :], = simulator.simulate(
        monitors=["agent_n"],
        record_per=MAX_EPOCH, max_epoch=MAX_EPOCH, seed=idx,
        n_key_states=N_KEY_STATES, dim_state=DIM_STATE, dim_action=DIM_ACTION,
        n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE,
        start_beta=START_BETA, last_beta=LAST_BETA)
  
  print('Start plotting figures for BLN')
  for idx, bln_exp in enumerate(BLN_EXPS):
    aa_seq_cdr3_abundance = extract_data(bln_exp)
    plot_simulation(axes[0][idx], agent_ns, np.sum(aa_seq_cdr3_abundance))
    population = plot_experiments(axes[0][idx], aa_seq_cdr3_abundance)

  print('Start plotting figures for PLN')
  for idx, pln_exp in enumerate(PLN_EXPS):
    aa_seq_cdr3_abundance = extract_data(pln_exp)
    plot_simulation(axes[1][idx], agent_ns, np.sum(aa_seq_cdr3_abundance))
    population = plot_experiments(axes[1][idx], aa_seq_cdr3_abundance)
  
  # Format figures
  format_figure(axes[0][0], ylabel='BLN clone size', labelbottom=False)
  format_figure(axes[0][1], ylabel='', labelbottom=False)
  format_figure(axes[0][2], ylabel='', labelbottom=False)
  format_figure(axes[1][0], ylabel='PLN clone size', labelbottom=True)
  format_figure(axes[1][1], ylabel='', labelbottom=True)
  format_figure(axes[1][2], ylabel='', labelbottom=True)
  fig.text(0.5, 0.03, 'Rank', ha='center', va='center')
  
  plt.savefig('fig4.eps')
  plt.close()

if __name__ == "__main__":
  # Experiments
  EXP_DATA_DIR = 'input'
  BLN_EXPS = ['SRR1188136', 'SRR1188167', 'SRR1188171']
  PLN_EXPS = ['SRR1188139', 'SRR1188142', 'SRR1188146']
  EXP_DATA_FILE = 'clones.txt'
  EXP_DELIMITER = '\t'
  
  # Simulations
  CACHE_DIR = '/tmp/fig4'
  N_SIM = 10
  N_HIDDEN = 5000
  N_KEY_STATES = 30
  DIM_STATE = 100
  DIM_ACTION = 20
  MAX_EPOCH = 100000
  LEARNING_RATE = 0.1
  START_BETA = 1.0
  LAST_BETA = 20

  main()
