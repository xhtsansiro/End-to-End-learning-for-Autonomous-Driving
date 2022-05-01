"""This script is used to plot the loss curves.

all five models in the same plot,
Comparing predicting three actions and predicting
only steering.
training loss in one plot and validation loss
in one plot.
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
# seed can only be 100,150 or 200
args = parser.parse_args()
# check if seed in [100,150,200]
if args.seed not in [100, 150, 200]:
    print("wrong seed number")
    sys.exit()

if args.cluster == 'yes':
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "09_Loss_plot/loss_comparison/"
    CNN_ALL = SDIR + 'seed' + str(args.seed) + "/cnn_all.csv"
    CNN_S = SDIR + 'seed' + str(args.seed) + "/cnn_steering.csv"
    LSTM_ALL = SDIR + 'seed' + str(args.seed) + "/lstm_all.csv"
    LSTM_S = SDIR + 'seed' + str(args.seed) + "/lstm_steering.csv"
    GRU_ALL = SDIR + 'seed' + str(args.seed) + "/gru_all.csv"
    GRU_S = SDIR + 'seed' + str(args.seed) + "/gru_steering.csv"
    CTGRU_ALL = SDIR + 'seed' + str(args.seed) + "/ctgru_all.csv"
    CTGRU_S = SDIR + 'seed' + str(args.seed) + "/ctgru_steering.csv"
    NCP_ALL = SDIR + 'seed' + str(args.seed) + "/ncp_all.csv"
    NCP_S = SDIR + 'seed' + str(args.seed) + "/ncp_steering.csv"
else:
    SDIR = "xxxx"   # local folder
    CNN_ALL = SDIR + 'seed' + str(args.seed) + "/cnn_all.csv"
    CNN_S = SDIR + 'seed' + str(args.seed) + "/cnn_steering.csv"
    LSTM_ALL = SDIR + 'seed' + str(args.seed) + "/lstm_all.csv"
    LSTM_S = SDIR + 'seed' + str(args.seed) + "/lstm_steering.csv"
    GRU_ALL = SDIR + 'seed' + str(args.seed) + "/gru_all.csv"
    GRU_S = SDIR + 'seed' + str(args.seed) + "/gru_steering.csv"
    CTGRU_ALL = SDIR + 'seed' + str(args.seed) + "/ctgru_all.csv"
    CTGRU_S = SDIR + 'seed' + str(args.seed) + "/ctgru_steering.csv"
    NCP_ALL = SDIR + 'seed' + str(args.seed) + "/ncp_all.csv"
    NCP_S = SDIR + 'seed' + str(args.seed) + "/ncp_steering.csv"

# cnn 3 action pairs
train_cnn_all = pd.read_csv(
    CNN_ALL,
    header=None)[1].values.tolist()[1:]
train_cnn_all = np.array(train_cnn_all).astype(dtype=np.float32).tolist()

valid_cnn_all = pd.read_csv(
    CNN_ALL,
    header=None)[2].values.tolist()[1:]
valid_cnn_all = np.array(valid_cnn_all).astype(dtype=np.float32).tolist()

# cnn only steering
train_cnn_s = pd.read_csv(
    CNN_S,
    header=None)[1].values.tolist()[1:]
train_cnn_s = np.array(train_cnn_s).astype(dtype=np.float32).tolist()

valid_cnn_s = pd.read_csv(
    CNN_S,
    header=None)[2].values.tolist()[1:]
valid_cnn_s = np.array(valid_cnn_s).astype(dtype=np.float32).tolist()

# cnn+lstm 3 action pairs
train_lstm_all = pd.read_csv(
    LSTM_ALL,
    header=None)[1].values.tolist()[1:]
train_lstm_all = np.array(train_lstm_all).astype(dtype=np.float32).tolist()

valid_lstm_all = pd.read_csv(
    LSTM_ALL,
    header=None)[2].values.tolist()[1:]
valid_lstm_all = np.array(valid_lstm_all).astype(dtype=np.float32).tolist()

# cnn+lstm steering
train_lstm_s = pd.read_csv(
    LSTM_S,
    header=None)[1].values.tolist()[1:]
train_lstm_s = np.array(train_lstm_s).astype(dtype=np.float32).tolist()

valid_lstm_s = pd.read_csv(
    LSTM_S,
    header=None)[2].values.tolist()[1:]
valid_lstm_s = np.array(valid_lstm_s).astype(dtype=np.float32).tolist()

# cnn+gru 3 action pairs
train_gru_all = pd.read_csv(
    GRU_ALL,
    header=None)[1].values.tolist()[1:]
train_gru_all = np.array(train_gru_all).astype(dtype=np.float32).tolist()

valid_gru_all = pd.read_csv(
    GRU_ALL,
    header=None)[2].values.tolist()[1:]
valid_gru_all = np.array(valid_gru_all).astype(dtype=np.float32).tolist()

# cnn+gru steering
train_gru_s = pd.read_csv(
    GRU_S,
    header=None)[1].values.tolist()[1:]
train_gru_s = np.array(train_gru_s).astype(dtype=np.float32).tolist()

valid_gru_s = pd.read_csv(
    GRU_S,
    header=None)[2].values.tolist()[1:]
valid_gru_s = np.array(valid_gru_s).astype(dtype=np.float32).tolist()

# cnn+ctgru 3 actions
train_ctgru_all = pd.read_csv(
    CTGRU_ALL,
    header=None)[1].values.tolist()[1:]
train_ctgru_all = np.array(train_ctgru_all).astype(dtype=np.float32).tolist()

valid_ctgru_all = pd.read_csv(
    CTGRU_ALL,
    header=None)[2].values.tolist()[1:]
valid_ctgru_all = np.array(valid_ctgru_all).astype(dtype=np.float32).tolist()

# cnn+ctgru steering
train_ctgru_s = pd.read_csv(
    CTGRU_S,
    header=None)[1].values.tolist()[1:]
train_ctgru_s = np.array(train_ctgru_s).astype(dtype=np.float32).tolist()

valid_ctgru_s = pd.read_csv(
    CTGRU_S,
    header=None)[2].values.tolist()[1:]
valid_ctgru_s = np.array(valid_ctgru_s).astype(dtype=np.float32).tolist()

# cnn+ncp 3 actions
train_ncp_all = pd.read_csv(
    NCP_ALL,
    header=None)[1].values.tolist()[1:]
train_ncp_all = np.array(train_ncp_all).astype(dtype=np.float32).tolist()

valid_ncp_all = pd.read_csv(
    NCP_ALL,
    header=None)[2].values.tolist()[1:]
valid_ncp_all = np.array(valid_ncp_all).astype(dtype=np.float32).tolist()

# cnn+ncp steering
train_ncp_s = pd.read_csv(
    NCP_S,
    header=None)[1].values.tolist()[1:]
train_ncp_s = np.array(train_ncp_s).astype(dtype=np.float32).tolist()

valid_ncp_s = pd.read_csv(
    NCP_S,
    header=None)[2].values.tolist()[1:]
valid_ncp_s = np.array(valid_ncp_s).astype(dtype=np.float32).tolist()


train_all = {'model_1': train_cnn_all,
             'model_2': train_lstm_all,
             'model_3': train_gru_all,
             'model_4': train_ctgru_all,
             'model_5': train_ncp_all}
train_s = {'model_1': train_cnn_s,
           'model_2': train_lstm_s,
           'model_3': train_gru_s,
           'model_4': train_ctgru_s,
           'model_5': train_ncp_s}

valid_all = {'model_1': valid_cnn_all,
             'model_2': valid_lstm_all,
             'model_3': valid_gru_all,
             'model_4': valid_ctgru_all,
             'model_5': valid_ncp_all}
valid_s = {'model_1': valid_cnn_s,
           'model_2': valid_lstm_s,
           'model_3': valid_gru_s,
           'model_4': valid_ctgru_s,
           'model_5': valid_ncp_s}

plt.clf()
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(2, 2, 1)
plt.title("Training loss curve of predicting all actions",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in train_all.items():
    ax1.plot(range(len(val)), val, label=key)
plt.legend()

ax2 = fig.add_subplot(2, 2, 3)
plt.title("Validation loss curve of predicting all actions",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in valid_all.items():
    ax2.plot(range(len(val)), val, label=key)
plt.legend()

ax3 = fig.add_subplot(2, 2, 2)
plt.title("Training loss curve of predicting steering",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in train_s.items():
    ax3.plot(range(len(val)), val, label=key)
plt.legend()

ax4 = fig.add_subplot(2, 2, 4)
plt.title("Validation loss curve of predicting steering",
          fontsize=11,
          fontweight='bold')
plt.xlabel("Number of Epoch", fontsize=11)
plt.ylabel("Mean-square error", fontsize=11)
plt.grid(ls='--')
for key, val in valid_s.items():
    ax4.plot(range(len(val)), val, label=key)
plt.legend()

fig.savefig(SDIR + str(args.seed) + "trainloss_summary.pdf")
