import sys
import numpy as np
from matplotlib import ticker
from matplotlib import pyplot as plt

sys.path.append("..")
from lib.sim import Simulator

# Run simulations
rewards_list = list()
epochs_list = list()
simulator = Simulator(cache_dir="/tmp/fig2")
for sim_idx in range(10):
  print("Start {} th simulation".format(sim_idx))
  epochs, rewards = simulator.simulate(
      monitors=["epochs", "rewards"],
      record_per=1000, max_epoch=100000, seed=sim_idx,
      n_key_states=30, dim_state=100, dim_action=20,
      n_hidden=5000, learning_rate=0.1, start_beta=1.0, last_beta=20)
  rewards_list.append(rewards)
rewards = np.array(rewards_list)

# Generate the figure
fig = plt.figure()
ax = fig.add_subplot(111)

low = np.percentile(rewards, 25, axis=0)
high = np.percentile(rewards, 75, axis=0)
median = np.median(rewards, axis=0)

ax.fill_between(epochs, high, low, color="#E6F1FF", alpha=0.1)
ax.plot(epochs, median, color="#6C9BD2")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_bounds(0, 10*10**4)
ax.set_xlim([-0.5*10**4, 10.5*10**4])
ax.spines["left"].set_bounds(0.35, 1.0)
ax.set_ylim([0.35, 1.00])

@ticker.FuncFormatter
def major_formatter(x, pos):
  num = int(x / 10**4)
  return "%d"%num

ax.xaxis.set_major_formatter(major_formatter)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

ax.set_xlabel(r"Training steps ($\times 10^4$)", fontsize=16)
ax.set_ylabel("Obtained reward per maximum", fontsize=16)

plt.savefig("fig2.eps")
plt.close()

