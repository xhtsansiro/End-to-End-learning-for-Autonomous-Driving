"""Analyze the deviation for CNN_head comparison.

Plot the distribution and boxplot of the deviation.
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
           "06_CNN_head_comparison/04_Comparison/" \
           "absolute_deviation_distribution/"
    PATH = SDIR + "deviation.csv"
else:  # in local PC,
    # define path and directory in local folder.
    pass

# read steering deviation from .csv file.
# NvidiaCNN+NCP
nvidia_s = pd.read_csv(PATH, header=None)[1].values.tolist()[1:]
nvidia_s = np.array(nvidia_s).astype(dtype=np.float32).tolist()
# AlexNet+NCP
alex_s = pd.read_csv(PATH, header=None)[2].values.tolist()[1:]
alex_s = np.array(alex_s).astype(dtype=np.float32).tolist()
# ResNet+NCP
res_s = pd.read_csv(PATH, header=None)[3].values.tolist()[1:]
res_s = np.array(res_s).astype(dtype=np.float32).tolist()

# read throttle deviation from .csv file.
# NvidiaCNN+NCP
nvidia_t = pd.read_csv(PATH, header=None)[4].values.tolist()[1:]
nvidia_t = np.array(nvidia_t).astype(dtype=np.float32).tolist()

# AlexNet+NCP
alex_t = pd.read_csv(PATH, header=None)[5].values.tolist()[1:]
alex_t = np.array(alex_t).astype(dtype=np.float32).tolist()

# ResNet+NCP
res_t = pd.read_csv(PATH, header=None)[6].values.tolist()[1:]
res_t = np.array(res_t).astype(dtype=np.float32).tolist()

# read brake deviation from .csv file.
# NvidiaCNN+NCP
nvidia_b = pd.read_csv(PATH, header=None)[7].values.tolist()[1:]
nvidia_b = np.array(nvidia_b).astype(dtype=np.float32).tolist()

# AlexNet+NCP
alex_b = pd.read_csv(PATH, header=None)[8].values.tolist()[1:]
alex_b = np.array(alex_b).astype(dtype=np.float32).tolist()

# ResNet+NCP
res_b = pd.read_csv(PATH, header=None)[9].values.tolist()[1:]
res_b = np.array(res_b).astype(dtype=np.float32).tolist()

# Histogram
# plot the deviation in hist. diagram
# each bin has interval 0.025
bin_s = np.arange(0, 0.425, 0.025).tolist()
bin_t = np.arange(0, 0.825, 0.025).tolist()
bin_b = np.arange(0, 1.025, 0.025).tolist()

#  show steering
plt.clf()

fig = plt.figure(figsize=(12, 11))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Steering deviation of NvidiaCNN+NCP',
              fontsize=12,
              fontweight='bold')
ax1.hist(nvidia_s, bins=bin_s, color='g')
ax1.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Steering deviation of AlexNet+NCP',
              fontsize=12,
              fontweight='bold')
ax2.hist(alex_s, bins=bin_s, color='brown')
ax2.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Steering deviation of ResNet+NCP',
              fontsize=12,
              fontweight='bold')
ax3.hist(res_s, bins=bin_s, color='b')
ax3.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')
fig.savefig(SDIR + "steering_dev_distribution.pdf")

#  show throttle
plt.clf()

fig = plt.figure(figsize=(12, 11))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Throttle deviation of NvidiaCNN+NCP',
              fontsize=12,
              fontweight='bold')
ax1.hist(nvidia_t, bins=bin_t, color='g')
ax1.set_ylim(0, 13000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Throttle deviation of AlexNet+NCP',
              fontsize=12,
              fontweight='bold')
ax2.hist(alex_t, bins=bin_t, color='brown')
ax2.set_ylim(0, 13000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Throttle deviation of ResNet+NCP',
              fontsize=12,
              fontweight='bold')
ax3.hist(res_t, bins=bin_t, color='b')
ax3.set_ylim(0, 13000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

fig.savefig(SDIR + "throttle_dev_distribution.pdf")

# show brake
plt.clf()

fig = plt.figure(figsize=(12, 11))
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Brake deviation of NvidiaCNN+NCP',
              fontsize=12,
              fontweight='bold')
ax1.hist(nvidia_b, bins=bin_b, color='g')
ax1.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Brake deviation of AlexNet+NCP',
              fontsize=12,
              fontweight='bold')
ax2.hist(alex_b, bins=bin_b, color='brown')
ax2.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

ax3 = fig.add_subplot(2, 2, 3)

ax3.set_title('Brake deviation of ResNet+NCP',
              fontsize=12,
              fontweight='bold')
ax3.hist(res_b, bins=bin_b, color='b')
ax3.set_ylim(0, 25000)
plt.xlabel('Absolute deviation', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.grid(ls='--')

fig.savefig(SDIR + "brake_dev_distribution.pdf")

# save the mean and variance of the deviation
mean_s_1 = round(np.mean(nvidia_s), 4)
var_s_1 = round(np.var(nvidia_s), 4)
# steering dev. mean of alex+NCP
mean_s_2 = round(np.mean(alex_s), 4)
var_s_2 = round(np.var(alex_s), 4)
# steering dev. mean of res+NCP
mean_s_3 = round(np.mean(res_s), 4)
var_s_3 = round(np.var(res_s), 4)

# throttle
mean_t_1 = round(np.mean(nvidia_t), 4)
var_t_1 = round(np.var(nvidia_t), 4)
# steering dev. mean of alex+NCP
mean_t_2 = round(np.mean(alex_t), 4)
var_t_2 = round(np.var(alex_t), 4)
# steering dev. mean of res+NCP
mean_t_3 = round(np.mean(res_t), 4)
var_t_3 = round(np.var(res_t), 4)

# brake
mean_b_1 = round(np.mean(nvidia_b), 4)
var_b_1 = round(np.var(nvidia_b), 4)
# steering dev. mean of alex+NCP
mean_b_2 = round(np.mean(alex_b), 4)
var_b_2 = round(np.var(alex_b), 4)
# steering dev. mean of res+NCP
mean_b_3 = round(np.mean(res_b), 4)
var_b_3 = round(np.var(res_b), 4)


F = SDIR + "mean&var.txt"
# output the information of transformation
S = "Steering mean:{},{},{}, variance:{},{},{}; ".format(
    mean_s_1, mean_s_2, mean_s_3, var_s_1, var_s_2, var_s_3)
T = "Throttle: mean:{},{},{}, variance:{},{},{}; ".format(
    mean_t_1, mean_t_2, mean_t_3, var_t_1, var_t_2, var_t_3)
B = "Brake: mean:{},{},{}, variance:{},{},{}; ".format(
    mean_b_1, mean_b_2, mean_b_3, var_b_1, var_b_2, var_b_3)
COMMENT = "the sequence is Nvidia_CNN+NCP, AlexNet+NCP, ResNet+NCP"


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
labels = 'NvidiaCNN+NCP', 'AlexNet+NCP', 'ResNet+NCP'
plt.boxplot([nvidia_s, alex_s, res_s],
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
labels = 'NvidiaCNN+NCP', 'AlexNet+NCP', 'ResNet+NCP'
plt.boxplot([nvidia_t, alex_t, res_t],
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
labels = 'NvidiaCNN+NCP', 'AlexNet+NCP', 'ResNet+NCP'
plt.boxplot([nvidia_b, alex_b, res_b],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR + "boxplot_brake.pdf")
