#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append("..")
from lib.sim import Simulator
from lib.utils import multinomial

# パラメータを設定
CACHE_DIR = "/tmp/fig5"
N_SAMP = 4000
N_PATHOGENS = [3, 10, 30, 100, 300]
N_HIDDEN = 5000
DIM_STATE = 100
DIM_ACTION = 20
MAX_EPOCH = 100000
LEARNING_RATE = 0.1
START_BETA = 1.0
LAST_BETA = 20.0

# 描画用のcanvasを用意
fig = plt.figure()
ax = fig.add_subplot(111)

# 描画用のpaletteを用意
palette = sns.color_palette("Blues", len(N_PATHOGENS))

# シミュレーションを実行
simulator = Simulator(CACHE_DIR)
for idx, n_pathogen in enumerate(N_PATHOGENS):
  print("Start {} th simulation".format(idx))
  agent_n, = simulator.simulate(
      monitors=["agent_n"],
      record_per=MAX_EPOCH, max_epoch=MAX_EPOCH, seed=idx,
      n_key_states=n_pathogen, dim_state=DIM_STATE, dim_action=DIM_ACTION,
      n_hidden=N_HIDDEN, learning_rate=LEARNING_RATE,
      start_beta=START_BETA, last_beta=LAST_BETA)
  nums = multinomial(agent_n, N_SAMP)
  ax.loglog(sorted(nums, reverse=True), label=str(n_pathogen), c=palette[idx])

# 画像の整形
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xlim([1, 4000])
ax.set_ylim([1, 40])
ax.set_yticks([1, 3, 10, 30])
ax.grid(True, "minor", "y", lw=0.5, c="#d0d0d0", alpha=0.5)
ax.grid(True, "major", "y", lw=0.5, c="#d0d0d0", alpha=0.5)
for tick in ax.yaxis.get_minor_ticks():
  tick.tick1On = False
for tick in ax.yaxis.get_major_ticks():
  tick.tick1On = False

ax.set_xlabel("Rank", fontsize=16)
ax.set_ylabel("The number of cells", fontsize=16)

leg = ax.legend()
leg.set_title("P")
plt.savefig("fig5.eps")
plt.close()
