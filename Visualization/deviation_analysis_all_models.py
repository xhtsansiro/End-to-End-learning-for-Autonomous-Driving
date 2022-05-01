"""Analyze the deviation for all models.

Plot the histogram and boxplot of the deviation.
models: CNN, CNN+LSTM, CNN+GRU, CNN+CTGRU, CNN+NCP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument("--cluster", type=str, default='no')
args = parser.parse_args()

if args.cluster == 'yes':  # in ssh terminal
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "08_Model_comparison/03_Comparison/"
    PATH = SDIR + "deviation.csv"
else:  # in local PC,
    # define path and directory in local folder.
    PATH = "/Users/haotianxue/Desktop/deviation.csv"
    SDIR = "/Users/haotianxue/Desktop/xxxx/"

# read steering deviation from .csv file.
# CNN
cnn_s = pd.read_csv(PATH, header=None)[1].values.tolist()[1:]
cnn_s = np.array(cnn_s).astype(dtype=np.float32).tolist()
# CNN+LSTM
lstm_s = pd.read_csv(PATH, header=None)[2].values.tolist()[1:]
lstm_s = np.array(lstm_s).astype(dtype=np.float32).tolist()
# CNN+GRU
gru_s = pd.read_csv(PATH, header=None)[3].values.tolist()[1:]
gru_s = np.array(gru_s).astype(dtype=np.float32).tolist()
# CNN+CTGRU
ctgru_s = pd.read_csv(PATH, header=None)[4].values.tolist()[1:]
ctgru_s = np.array(ctgru_s).astype(dtype=np.float32).tolist()
# CNN+NCP
ncp_s = pd.read_csv(PATH, header=None)[5].values.tolist()[1:]
ncp_s = np.array(ncp_s).astype(dtype=np.float32).tolist()

# read throttle deviation from .csv file.
# CNN
cnn_t = pd.read_csv(PATH, header=None)[6].values.tolist()[1:]
cnn_t = np.array(cnn_t).astype(dtype=np.float32).tolist()
# CNN+LSTM
lstm_t = pd.read_csv(PATH, header=None)[7].values.tolist()[1:]
lstm_t = np.array(lstm_t).astype(dtype=np.float32).tolist()
# CNN+GRU
gru_t = pd.read_csv(PATH, header=None)[8].values.tolist()[1:]
gru_t = np.array(gru_t).astype(dtype=np.float32).tolist()
# CNN+CTGRU
ctgru_t = pd.read_csv(PATH, header=None)[9].values.tolist()[1:]
ctgru_t = np.array(ctgru_t).astype(dtype=np.float32).tolist()
# CNN+NCP
ncp_t = pd.read_csv(PATH, header=None)[10].values.tolist()[1:]
ncp_t = np.array(ncp_t).astype(dtype=np.float32).tolist()

# read brake deviation from .csv file.
# CNN
cnn_b = pd.read_csv(PATH, header=None)[11].values.tolist()[1:]
cnn_b = np.array(cnn_b).astype(dtype=np.float32).tolist()
# CNN+LSTM
lstm_b = pd.read_csv(PATH, header=None)[12].values.tolist()[1:]
lstm_b = np.array(lstm_b).astype(dtype=np.float32).tolist()
# CNN+GRU
gru_b = pd.read_csv(PATH, header=None)[13].values.tolist()[1:]
gru_b = np.array(gru_b).astype(dtype=np.float32).tolist()
# CNN+CTGRU
ctgru_b = pd.read_csv(PATH, header=None)[14].values.tolist()[1:]
ctgru_b = np.array(ctgru_b).astype(dtype=np.float32).tolist()
# CNN+NCP
ncp_b = pd.read_csv(PATH, header=None)[15].values.tolist()[1:]
ncp_b = np.array(ncp_b).astype(dtype=np.float32).tolist()

# Histogram
# plot the deviation in hist. diagram
# each bin has interval 0.025
bin_s = np.arange(0, 0.425, 0.025).tolist()
bin_t = np.arange(0, 0.825, 0.025).tolist()
bin_b = np.arange(0, 1.025, 0.025).tolist()

#  show steering
plt.clf()

fig = plt.figure(figsize=(12, 16.5))
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_title('Steering deviation of CNN',
              fontsize=12,
              fontweight='bold')
ax1.hist(cnn_s, bins=bin_s, color='g')
ax1.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(3, 2, 2)
ax2.set_title('Steering deviation of CNN+LSTM',
              fontsize=12,
              fontweight='bold')

ax2.hist(lstm_s, bins=bin_s, color='brown')
ax2.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(3, 2, 3)
ax3.set_title('Steering deviation of CNN+GRU',
              fontsize=12,
              fontweight='bold')
ax3.hist(gru_s, bins=bin_s, color='b')
ax3.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax4 = fig.add_subplot(3, 2, 4)
ax4.set_title('Steering deviation of CNN+CT-GRU',
              fontsize=13,
              fontweight='bold')
ax4.hist(ctgru_s, bins=bin_s, color='violet')
ax4.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax5 = fig.add_subplot(3, 2, 5)
ax5.set_title('Steering deviation of CNN+NCP',
              fontsize=12,
              fontweight='bold')
ax5.hist(ncp_s, bins=bin_s, color='orange')
ax5.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')
fig.savefig(SDIR + "steering_dev_distribution.pdf")

#  show throttle
plt.clf()

fig = plt.figure(figsize=(12, 16.5))
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_title('Throttle deviation of CNN',
              fontsize=12,
              fontweight='bold')
ax1.hist(cnn_t, bins=bin_t, color='g')
ax1.set_ylim(0, 13000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(3, 2, 2)
ax2.set_title('Throttle deviation of CNN+LSTM',
              fontsize=12,
              fontweight='bold')
ax2.hist(lstm_t, bins=bin_t, color='brown')
ax2.set_ylim(0, 13000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(3, 2, 3)
ax3.set_title('Throttle deviation of CNN+GRU',
              fontsize=12,
              fontweight='bold')
ax3.hist(gru_t, bins=bin_t, color='b')
ax3.set_ylim(0, 13000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax4 = fig.add_subplot(3, 2, 4)
ax4.set_title('Throttle deviation of CNN+CT-GRU',
              fontsize=12,
              fontweight='bold')
ax4.hist(ctgru_t, bins=bin_t, color='violet')
ax4.set_ylim(0, 13000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax5 = fig.add_subplot(3, 2, 5)
ax5.set_title('Throttle deviation of CNN+NCP',
              fontsize=12,
              fontweight='bold')
ax5.hist(ncp_t, bins=bin_t, color='orange')
ax5.set_ylim(0, 13000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

fig.savefig(SDIR + "throttle_dev_distribution.pdf")

# show brake
plt.clf()

fig = plt.figure(figsize=(12, 16.5))
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_title('Brake deviation of CNN',
              fontsize=12,
              fontweight='bold')
ax1.hist(cnn_b, bins=bin_b, color='g')
ax1.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(3, 2, 2)
ax2.set_title('Brake deviation of CNN+LSTM',
              fontsize=12,
              fontweight='bold')
ax2.hist(lstm_b, bins=bin_b, color='brown')
ax2.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(3, 2, 3)
ax3.set_title('Brake deviation of CNN+GRU',
              fontsize=12,
              fontweight='bold')
ax3.hist(gru_b, bins=bin_b, color='b')
ax3.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax4 = fig.add_subplot(3, 2, 4)
ax4.set_title('Brake deviation of CNN+CT-GRU',
              fontsize=12,
              fontweight='bold')
ax4.hist(ctgru_b, bins=bin_b, color='violet')
ax4.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

ax5 = fig.add_subplot(3, 2, 5)
ax5.set_title('Brake deviation of CNN+NCP',
              fontsize=12,
              fontweight='bold')
ax5.hist(ncp_b, bins=bin_b, color='orange')
ax5.set_ylim(0, 25000)
plt.xlabel("Absolute deviation", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.grid(ls='--')

fig.savefig(SDIR + "brake_dev_distribution.pdf")

# save the mean and variance of the deviation
mean_s_1 = round(np.mean(cnn_s), 4)
var_s_1 = round(np.var(cnn_s), 4)

mean_s_2 = round(np.mean(lstm_s), 4)
var_s_2 = round(np.var(lstm_s), 4)

mean_s_3 = round(np.mean(gru_s), 4)
var_s_3 = round(np.var(gru_s), 4)

mean_s_4 = round(np.mean(ctgru_s), 4)
var_s_4 = round(np.var(ctgru_s), 4)

mean_s_5 = round(np.mean(ncp_s), 4)
var_s_5 = round(np.var(ncp_s), 4)

# throttle
mean_t_1 = round(np.mean(cnn_t), 4)
var_t_1 = round(np.var(cnn_t), 4)

mean_t_2 = round(np.mean(lstm_t), 4)
var_t_2 = round(np.var(lstm_t), 4)

mean_t_3 = round(np.mean(gru_t), 4)
var_t_3 = round(np.var(gru_t), 4)

mean_t_4 = round(np.mean(ctgru_t), 4)
var_t_4 = round(np.var(ctgru_t), 4)

mean_t_5 = round(np.mean(ncp_t), 4)
var_t_5 = round(np.var(ncp_t), 4)

# brake
mean_b_1 = round(np.mean(cnn_b), 4)
var_b_1 = round(np.var(cnn_b), 4)

mean_b_2 = round(np.mean(lstm_b), 4)
var_b_2 = round(np.var(lstm_b), 4)

mean_b_3 = round(np.mean(gru_b), 4)
var_b_3 = round(np.var(gru_b), 4)

mean_b_4 = round(np.mean(ctgru_b), 4)
var_b_4 = round(np.var(ctgru_b), 4)

mean_b_5 = round(np.mean(ncp_b), 4)
var_b_5 = round(np.var(ncp_b), 4)


F = SDIR + "mean&var.txt"
# output the information of transformation
S = "Steering mean:{},{},{},{},{}, variance:{},{},{},{},{}; ".format(
    mean_s_1, mean_s_2, mean_s_3, mean_s_4, mean_s_5,
    var_s_1, var_s_2, var_s_3, var_s_4, var_s_5)
T = "Throttle: mean:{},{},{},{},{}, variance:{},{},{},{},{}; ".format(
    mean_t_1, mean_t_2, mean_t_3, mean_t_4, mean_t_5,
    var_t_1, var_t_2, var_t_3, var_t_4, var_t_5)
B = "Brake: mean:{},{},{},{},{}, variance:{},{},{},{},{}; ".format(
    mean_b_1, mean_b_2, mean_b_3, mean_b_4, mean_b_5,
    var_b_1, var_b_2, var_b_3, var_b_4, var_b_5)
COMMENT = "the sequence is CNN, CNN+LSTM, CNN+GRU, CNN+CT-GRU," \
          "CNN+NCP"


with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(S)
    fil.write(T)
    fil.write(B)
    fil.write(COMMENT)

# Boxplot
# plot deviation of steering
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering",
          fontsize=11,
          fontweight='bold')
labels = 'CNN', 'CNN+LSTM', 'CNN+GRU', 'CNN+CT-GRU', 'CNN+NCP'
plt.boxplot([cnn_s, lstm_s, gru_s, ctgru_s, ncp_s],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR + "boxplot_steering.pdf")

# plot deviation of throttle
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of throttle",
          fontsize=11,
          fontweight='bold')
labels = 'CNN', 'CNN+LSTM', 'CNN+GRU', 'CNN+CT-GRU', 'CNN+NCP'
plt.boxplot([cnn_t, lstm_t, gru_t, ctgru_t, ncp_t],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR + "boxplot_throttle.pdf")

# plot deviation of brake
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of brake",
          fontsize=11,
          fontweight='bold')
labels = 'CNN', 'CNN+LSTM', 'CNN+GRU', 'CNN+CT-GRU', 'CNN+NCP'
plt.boxplot([cnn_b, lstm_b, gru_b, ctgru_b, ncp_b],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR + "boxplot_brake.pdf")
