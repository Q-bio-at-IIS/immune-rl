#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("..")
from lib.sim import Simulator

# 描画用の図を二つ用意する
_, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Simulatorを用意する
simulator = Simulator(cache_dir="/tmp/fig3")

# 細胞数分布の時間発展を描画する
print("Start 0 th simulation")
epochs, agent_ns = simulator.simulate(
    monitors=["epochs", "agent_ns"],
    record_per=1000, max_epoch=100000, seed=0,
    n_key_states=30, dim_state=100, dim_action=20,
    n_hidden=5000, learning_rate=0.1, start_beta=1.0, last_beta=20)
ax1.semilogy(epochs / 10**4, agent_ns, linewidth=0.1)
ax1.set_ylim([1, 30])

# Rank distribution を描画する
for idx in range(1, 100):
  print("Start {} th simulation".format(idx))
  agent_n, = simulator.simulate(
    monitors=["agent_n"],
    record_per=100000, max_epoch=100000, seed=idx,
    n_key_states=30, dim_state=100, dim_action=20,
    n_hidden=5000, learning_rate=0.1, start_beta=1.0, last_beta=20
  )
  ax2.loglog(sorted(agent_n, reverse=True), color='#d0d0d0', linewidth=0.2)
ax2.loglog(sorted(agent_ns[-1], reverse=True), color='#4C72B0')
ax2.set_ylim([1, 30])

# プロットの整形
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
for tick in ax1.yaxis.get_minor_ticks():
  tick.tick1On = False
for tick in ax1.yaxis.get_major_ticks():
  tick.tick1On = False
for tick in ax2.yaxis.get_minor_ticks():
  tick.tick1On = False
for tick in ax2.yaxis.get_major_ticks():
  tick.tick1On = False
ax1.tick_params(axis='y', which='both', top=False,
    labelbottom=True, left=False, right=False, labelleft=True)
ax2.tick_params(axis='y', which='both', top=False,
    labelbottom=True, left=False, right=False, labelleft=False)
ax1.grid(True, "minor", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
ax1.grid(True, "major", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
ax2.grid(True, "minor", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
ax2.grid(True, "major", "y", lw=0.5, c="#d0d0d0", alpha=0.8)
ax1.set_xlabel(r"Training steps ($\times 10^4$)")
ax1.set_ylabel("Clone size")
ax2.set_xlabel("Rank")
plt.savefig('fig3.eps')
plt.close()
