"""This script plots the distribution of actions for LGSVL Data."""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--cluster", type=str, default='no')

args = parser.parse_args()

if args.cluster == 'yes':
    train_name = "/home/ubuntu/repos/LGSVL/train_data"
    valid_name = "/home/ubuntu/repos/LGSVL/valid_data"
    SDIR = "/home/ubuntu/repos/results_haotian/02_LGSVL_Data/" \
           "07_Action_distribution/"
else:
    train_name = "/Volumes/ftm/Projekte/Studenten_Betz/EventScape/train_data"
    valid_name = "/Volumes/ftm/Projekte/Studenten_Betz/EventScape/valid_data"
    SDIR = "/Volumes/ftm/Projekte/Studenten_Betz/expert/results_haotian/" \
           "02_LGSVL_Data/07_Action_distribution/"

steering_train, throttle_train, brake_train = [], [], []
steering_valid, throttle_valid, brake_valid = [], [], []

data_town_train = sorted(os.listdir(train_name))
data_town_train = \
    list(filter(lambda a: '.DS_Store' not in a, data_town_train))
# filter the .DS_Store
data_town_train = \
    list(filter(lambda a: '._Town' not in a, data_town_train))
# filter double file created in MacOS
data_town_train = \
    list(filter(lambda a: 'result' not in a, data_town_train))
# filter double file created in MacOS

for i in data_town_train:

    subfolder = os.path.join(train_name, i)
    sequences = sorted(os.listdir(subfolder))
    sequences = list(filter(lambda a: '.DS_Store' not in a, sequences))
    # filter the .DS_Store

    for sequence in sequences:
        sequence_path = os.path.join(subfolder, sequence)
        steering_path = os.path.join(sequence_path, 'action/steering.csv')
        throttle_path = os.path.join(sequence_path, 'action/throttle.csv')
        brake_path = os.path.join(sequence_path, 'action/brake.csv')
        steering_raw = pd.read_csv(steering_path,
                                   header=None)[1].values.tolist()
        throttle_raw = pd.read_csv(throttle_path,
                                   header=None)[1].values.tolist()
        brake_raw = pd.read_csv(brake_path, header=None)[1].values.tolist()

        steering_train.extend(steering_raw)
        throttle_train.extend(throttle_raw)
        brake_train.extend(brake_raw)


data_town_valid = sorted(os.listdir(valid_name))
data_town_valid = \
    list(filter(lambda a: '.DS_Store' not in a, data_town_valid))
# filter the .DS_Store
data_town_valid = \
    list(filter(lambda a: '._Town' not in a, data_town_valid))
# filter double file created in MacOS
data_town_valid = \
    list(filter(lambda a: 'result' not in a, data_town_valid))
# filter double file created in MacO

for i in data_town_valid:

    subfolder = os.path.join(valid_name, i)
    sequences = sorted(os.listdir(subfolder))
    sequences = list(filter(lambda a: '.DS_Store' not in a, sequences))
    # filter the .DS_Store

    for sequence in sequences:
        sequence_path = os.path.join(subfolder, sequence)
        steering_path = os.path.join(sequence_path, 'action/steering.csv')
        throttle_path = os.path.join(sequence_path, 'action/throttle.csv')
        brake_path = os.path.join(sequence_path, 'action/brake.csv')
        steering_raw = pd.read_csv(steering_path,
                                   header=None)[1].values.tolist()
        throttle_raw = pd.read_csv(throttle_path,
                                   header=None)[1].values.tolist()
        brake_raw = pd.read_csv(brake_path,
                                header=None)[1].values.tolist()

        steering_valid.extend(steering_raw)
        throttle_valid.extend(throttle_raw)
        brake_valid.extend(brake_raw)

# calculate variance of steering for train and test data
steering_train_var = round(np.var(steering_train), 3)
steering_train_mean = round(np.mean(steering_train), 3)
steering_valid_var = round(np.var(steering_valid), 3)
steering_valid_mean = round(np.mean(steering_valid), 3)

# calculate variance of throttle for train and test data
throttle_train_var = round(np.var(throttle_train), 3)
throttle_train_mean = round(np.mean(throttle_train), 3)
throttle_valid_var = round(np.var(throttle_valid), 3)
throttle_valid_mean = round(np.mean(throttle_valid), 3)

# calculate variance of brake for train and test data
brake_train_var = round(np.var(brake_train), 3)
brake_train_mean = round(np.mean(brake_train), 3)
brake_valid_var = round(np.var(brake_valid), 3)
brake_valid_mean = round(np.mean(brake_valid), 3)

# start plotting

bin_s = np.arange(-0.50, 0.525, 0.025).tolist()
bin_t = np.arange(0, 1.025, 0.025).tolist()
bin_b = np.arange(0, 1.025, 0.025).tolist()

# steering
plt.clf()

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('LGSVL: steering distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(steering_train, bins=bin_s, color='g')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')


ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('LGSVL: steering distribution of validation data',
              fontsize=11,
              fontweight='bold')
ax2.hist(steering_valid, bins=bin_s, color='brown')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
fig.savefig(SDIR + "/steering_distribution.pdf")

# throttle
plt.clf()

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('LGSVL: throttle distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(throttle_train, bins=bin_t, color='g')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('LGSVL: throttle distribution of validation data',
              fontsize=11,
              fontweight='bold')
ax2.hist(throttle_valid, bins=bin_t, color='brown')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
fig.savefig(SDIR + "/throttle_distribution.pdf")

# brake
plt.clf()

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('LGSVL: brake distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(brake_train, bins=bin_b, color='g')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
# plt.text(0.65, 130000, 'mean: {}, var: {}'.format(
#    brake_train_mean, brake_train_var), fontsize=6)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('LGSVL: brake distribution of validation data',
              fontsize=11,
              fontweight='bold')
ax2.hist(brake_valid, bins=bin_b, color='brown')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
fig.savefig(SDIR + "/brake_distribution.pdf")

F = SDIR + "/mean&var.txt"
# output the information of transformation
S = "Steering mean:{},{}, variance:{},{}; ".format(
    steering_train_mean, steering_valid_mean,
    steering_train_var, steering_valid_var)
T = "Throttle: mean:{},{}, variance:{},{}; ".format(
    throttle_train_mean, throttle_valid_mean,
    throttle_train_var, throttle_valid_var)
B = "Brake: mean:{},{}, variance:{},{}; ".format(
    brake_train_mean, brake_valid_mean,
    brake_train_var, brake_valid_var)
COMMENT = "the sequence is training dataset, validation dataset"


with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(S)
    fil.write(T)
    fil.write(B)
    fil.write(COMMENT)

