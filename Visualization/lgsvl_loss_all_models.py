"""This script is used to plot the loss curves.

all five models in the same plot, training loss
in one plot and validation loss in one plot.
Five models are CNN, CNN+LSTM, CNN+GRU, CNN+CTGRU,
and CNN+NCP
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
# seed can only be 100,150 or 200.

if args.cluster == 'yes':
    SDIR = "/home/ubuntu/repos/results_haotian/02_LGSVL_Data/" \
           "08_Loss_plot/loss_all_models/"
    PATH_CNN = SDIR + 'cnn_seed' + str(args.seed) + ".csv"
    PATH_LSTM = SDIR + 'cnn+lstm_seed' + str(args.seed) + ".csv"
    PATH_GRU = SDIR + 'cnn+gru_seed' + str(args.seed) + ".csv"
    PATH_CTGRU = SDIR + 'cnn+ctgru_seed' + str(args.seed) + ".csv"
    PATH_NCP = SDIR + 'cnn+ncp_seed' + str(args.seed) + ".csv"
else:
    SDIR = "/Users/haotianxue/Desktop/loss_lgsvl/"
    PATH_CNN = SDIR + 'cnn_seed' + str(args.seed) + ".csv"
    PATH_LSTM = SDIR + 'cnn+lstm_seed' + str(args.seed) + ".csv"
    PATH_GRU = SDIR + 'cnn+gru_seed' + str(args.seed) + ".csv"
    PATH_CTGRU = SDIR + 'cnn+ctgru_seed' + str(args.seed) + ".csv"
    PATH_NCP = SDIR + 'cnn+ncp_seed' + str(args.seed) + ".csv"

# cnn
train_cnn = pd.read_csv(
    PATH_CNN,
    header=None)[1].values.tolist()[1:]
train_cnn = np.array(train_cnn).astype(dtype=np.float32).tolist()

valid_cnn = pd.read_csv(
    PATH_CNN,
    header=None)[2].values.tolist()[1:]
valid_cnn = np.array(valid_cnn).astype(dtype=np.float32).tolist()

# cnn+lstm
train_lstm = pd.read_csv(
    PATH_LSTM,
    header=None)[1].values.tolist()[1:]
train_lstm = np.array(train_lstm).astype(dtype=np.float32).tolist()

valid_lstm = pd.read_csv(
    PATH_LSTM,
    header=None)[2].values.tolist()[1:]
valid_lstm = np.array(valid_lstm).astype(dtype=np.float32).tolist()

# cnn+gru
train_gru = pd.read_csv(
    PATH_GRU,
    header=None)[1].values.tolist()[1:]
train_gru = np.array(train_gru).astype(dtype=np.float32).tolist()

valid_gru = pd.read_csv(
    PATH_GRU,
    header=None)[2].values.tolist()[1:]
valid_gru = np.array(valid_gru).astype(dtype=np.float32).tolist()

# cnn+ctgru
train_ctgru = pd.read_csv(
    PATH_CTGRU,
    header=None)[1].values.tolist()[1:]
train_ctgru = np.array(train_ctgru).astype(dtype=np.float32).tolist()

valid_ctgru = pd.read_csv(
    PATH_CTGRU,
    header=None)[2].values.tolist()[1:]
valid_ctgru = np.array(valid_ctgru).astype(dtype=np.float32).tolist()

# cnn+ncp
train_ncp = pd.read_csv(
    PATH_NCP,
    header=None)[1].values.tolist()[1:]
train_ncp = np.array(train_ncp).astype(dtype=np.float32).tolist()

valid_ncp = pd.read_csv(
    PATH_NCP,
    header=None)[2].values.tolist()[1:]
valid_ncp = np.array(valid_ncp).astype(dtype=np.float32).tolist()


data_train = {'model_1': train_cnn,
              'model_2': train_lstm,
              'model_3': train_gru,
              'model_4': train_ctgru,
              'model_5': train_ncp}
data_valid = {'model_1': valid_cnn,
              'model_2': valid_lstm,
              'model_3': valid_gru,
              'model_4': valid_ctgru,
              'model_5': valid_ncp}

plt.clf()
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
plt.title("Training loss curve of all models",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in data_train.items():
    ax1.plot(range(len(val)), val, label=key)
plt.legend()

ax2 = fig.add_subplot(1, 2, 2)
plt.title("Validation loss curve of all models",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in data_valid.items():
    ax2.plot(range(len(val)), val, label=key)
plt.legend()

fig.savefig(SDIR + str(args.seed) + "trainloss_summary.pdf")
