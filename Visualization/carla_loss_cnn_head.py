"""This script is used to plot loss curves.

in the same plot, models are:NvidiaCNN+NCP,
ResNet+NCP, AlexNet+NCP.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--cluster", type=str, default='no')
parser.add_argument("--seed", type=int, default=200)

args = parser.parse_args()

if args.seed not in [100, 150, 200]:
    print("wrong seed number")
    sys.exit()
#  seed can only be 100,150 or 200.

if args.cluster == 'yes':
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "09_Loss_plot/loss_cnnhead_comparison/"
    if args.seed == 100:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    elif args.seed == 150:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    elif args.seed == 200:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    else:
        print("Wrong seed number.")
        sys.exit()
else:
    SDIR = "xxxxx"  # local folder where data is saved
    if args.seed == 100:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    elif args.seed == 150:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    elif args.seed == 200:
        PATH = SDIR + 'seed' + str(args.seed) + "/"
    else:
        print("Wrong seed number.")
        sys.exit()

# nvidia
train_n = pd.read_csv(
    PATH+"nvidia.csv",
    header=None)[1].values.tolist()[1:]
train_n = np.array(train_n).astype(dtype=np.float32).tolist()

valid_n = pd.read_csv(
    PATH+"nvidia.csv",
    header=None)[2].values.tolist()[1:]
valid_n = np.array(valid_n).astype(dtype=np.float32).tolist()

# alexNet
train_a = pd.read_csv(
    PATH+"alex.csv",
    header=None)[1].values.tolist()[1:]
train_a = np.array(train_a).astype(dtype=np.float32).tolist()

valid_a = pd.read_csv(
    PATH+"alex.csv",
    header=None)[2].values.tolist()[1:]
valid_a = np.array(valid_a).astype(dtype=np.float32).tolist()

# ResNet
train_r = pd.read_csv(
    PATH+"res.csv",
    header=None)[1].values.tolist()[1:]
train_r = np.array(train_r).astype(dtype=np.float32).tolist()

valid_r = pd.read_csv(
    PATH+"res.csv",
    header=None)[2].values.tolist()[1:]
valid_r = np.array(valid_r).astype(dtype=np.float32).tolist()
data_train = {'model_1': train_n,
              'model_2': train_a,
              'model_3': train_r}
data_valid = {'model_1': valid_n,
              'model_2': valid_a,
              'model_3': valid_r}

plt.clf()
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
plt.title("Training loss curve of CNN+NCP",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in data_train.items():
    ax1.plot(range(len(val)), val, label=key)
plt.legend()

ax2 = fig.add_subplot(1, 2, 2)
plt.title("Validation loss curve of CNN+NCP",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in data_valid.items():
    ax2.plot(range(len(val)), val, label=key)
plt.legend()

fig.savefig(SDIR + str(args.seed) + "trainloss_summary.pdf")
