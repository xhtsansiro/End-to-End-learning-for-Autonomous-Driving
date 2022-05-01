"""This script plots boxplot for all 5 neural networks.

Boxplot of steering for models predicting all actions
Boxplot of steering for models predicting only steering.
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
    SDIR_S = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "08_Model_comparison/04_Steering/"
    PATH_S = SDIR_S + "deviation.csv"

    SDIR_A = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
             "08_Model_comparison/03_Comparison/"
    PATH_A = SDIR_A + "deviation.csv"
else:  # in local PC,
    # define path and directory in local folder.
    pass

# CNN
cnn_s_1 = pd.read_csv(PATH_A, header=None)[1].values.tolist()[1:]
cnn_s_1 = np.array(cnn_s_1).astype(dtype=np.float32).tolist()
# CNN+LSTM
lstm_s_1 = pd.read_csv(PATH_A, header=None)[2].values.tolist()[1:]
lstm_s_1 = np.array(lstm_s_1).astype(dtype=np.float32).tolist()
# CNN+GRU
gru_s_1 = pd.read_csv(PATH_A, header=None)[3].values.tolist()[1:]
gru_s_1 = np.array(gru_s_1).astype(dtype=np.float32).tolist()
# CNN+CTGRU
ctgru_s_1 = pd.read_csv(PATH_A, header=None)[4].values.tolist()[1:]
ctgru_s_1 = np.array(ctgru_s_1).astype(dtype=np.float32).tolist()
# CNN+NCP
ncp_s_1 = pd.read_csv(PATH_A, header=None)[5].values.tolist()[1:]
ncp_s_1 = np.array(ncp_s_1).astype(dtype=np.float32).tolist()


# CNN
cnn_s_2 = pd.read_csv(PATH_S, header=None)[1].values.tolist()[1:]
cnn_s_2 = np.array(cnn_s_2).astype(dtype=np.float32).tolist()
# CNN+LSTM
lstm_s_2 = pd.read_csv(PATH_S, header=None)[2].values.tolist()[1:]
lstm_s_2 = np.array(lstm_s_2).astype(dtype=np.float32).tolist()
# CNN+GRU
gru_s_2 = pd.read_csv(PATH_S, header=None)[3].values.tolist()[1:]
gru_s_2 = np.array(gru_s_2).astype(dtype=np.float32).tolist()
# CNN+CTGRU
ctgru_s_2 = pd.read_csv(PATH_S, header=None)[4].values.tolist()[1:]
ctgru_s_2 = np.array(ctgru_s_2).astype(dtype=np.float32).tolist()
# CNN+NCP
ncp_s_2 = pd.read_csv(PATH_S, header=None)[5].values.tolist()[1:]
ncp_s_2 = np.array(ncp_s_2).astype(dtype=np.float32).tolist()

# Boxplot
# plot deviation of steering CNN
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering of CNN",
          fontsize=11,
          fontweight='bold')
labels = 'Model_1', 'Model_2'
plt.boxplot([cnn_s_1, cnn_s_2],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR_S + "boxplot_steering_CNN.pdf")

# Boxplot
# plot deviation of steering CNN+LSTM
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering of CNN+LSTM",
          fontsize=11,
          fontweight='bold')
labels = 'Model_1', 'Model_2'
plt.boxplot([lstm_s_1, lstm_s_2],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')

plt.savefig(SDIR_S + "boxplot_steering_CNN+LSTM.pdf")

# Boxplot
# plot deviation of steering CNN+GRU
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering of CNN+GRU",
          fontsize=11,
          fontweight='bold')
labels = 'Model_1', 'Model_2'
plt.boxplot([gru_s_1, gru_s_2],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')
plt.savefig(SDIR_S + "boxplot_steering_CNN+GRU.pdf")

# Boxplot
# plot deviation of steering CNN+CTGRU
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering of CNN+CT-GRU",
          fontsize=11,
          fontweight='bold')
labels = 'Model_1', 'Model_2'
plt.boxplot([ctgru_s_1, ctgru_s_2],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')
plt.savefig(SDIR_S + "boxplot_steering_CNN+CTGRU.pdf")

# Boxplot
# plot deviation of steering CNN+NCP
plt.clf()
plt.figure(figsize=(6, 3))
plt.title("Absolute deviation of steering of CNN+NCP",
          fontsize=11,
          fontweight='bold')
labels = 'Model_1', 'Model_2'
plt.boxplot([ncp_s_1, ncp_s_2],
            labels=labels,
            showmeans=True,
            showfliers=False)
plt.ylabel('Absolute deviation', fontsize=11)
plt.grid(ls='--')
plt.savefig(SDIR_S + "boxplot_steering_CNN+NCP.pdf")
