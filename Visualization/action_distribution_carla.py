"""This script aims to check the action distributions in CARLA Data."""

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
    TRAIN_NAME = "/home/ubuntu/repos/EventScape/train_data"
    VALID_NAME = "/home/ubuntu/repos/EventScape/valid_data"
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "07_Action_distribution/"
else:
    TRAIN_NAME = "/Volumes/ftm/Projekte/Studenten_Betz/EventScape/train_data"
    VALID_NAME = "/Volumes/ftm/Projekte/Studenten_Betz/EventScape/valid_data"
    SDIR = "/Volumes/ftm/Projekte/Studenten_Betz/expert/results_haotian/" \
           "01_CARLA_Data/07_Action_distribution/"

steering_train, throttle_train, brake_train = [], [], []
steering_valid, throttle_valid, brake_valid = [], [], []

data_town_train = sorted(os.listdir(TRAIN_NAME))
data_town_train = list(filter(lambda a: '.DS_Store' not in a, data_town_train))
# filter the .DS_Store
data_town_train = list(filter(lambda a: '._Town' not in a, data_town_train))
# filter double file created in MacOS
data_town_train = list(filter(lambda a: 'result' not in a, data_town_train))
# filter double file created in MacOS

for i in data_town_train:

    subfolder = os.path.join(TRAIN_NAME, i)
    sequences = sorted(os.listdir(subfolder))
    sequences = list(filter(lambda a: '.DS_Store' not in a, sequences))
    # filter the .DS_Store

    for sequence in sequences:
        sequence_path = os.path.join(subfolder, sequence)
        steering_path = os.path.join(sequence_path,
                                     'vehicle_data/steering.txt')
        throttle_path = os.path.join(sequence_path,
                                     'vehicle_data/throttle.txt')
        brake_path = os.path.join(sequence_path,
                                  'vehicle_data/brake.txt')
        steering_raw = pd.read_csv(steering_path,
                                   header=None)[0].values.tolist()
        throttle_raw = pd.read_csv(throttle_path,
                                   header=None)[0].values.tolist()
        brake_raw = pd.read_csv(brake_path,
                                header=None)[0].values.tolist()

        # take one sample every 40 samples,
        for number, item in enumerate(steering_raw):
            if number % 40 == 0:
                steering_train.append(item)

        for number, item in enumerate(throttle_raw):
            if number % 40 == 0:
                throttle_train.append(item)

        for number, item in enumerate(brake_raw):
            if number % 40 == 0:
                brake_train.append(item)


data_town_valid = sorted(os.listdir(VALID_NAME))
data_town_valid = list(filter(lambda a: '.DS_Store' not in a, data_town_valid))
# filter the .DS_Store
data_town_valid = list(filter(lambda a: '._Town' not in a, data_town_valid))
# filter double file created in MacOS
data_town_valid = list(filter(lambda a: 'result' not in a, data_town_valid))
# filter double file created in MacO

for i in data_town_valid:

    subfolder = os.path.join(VALID_NAME, i)
    sequences = sorted(os.listdir(subfolder))
    sequences = list(filter(lambda a: '.DS_Store' not in a, sequences))
    # filter the .DS_Store

    for sequence in sequences:
        sequence_path = os.path.join(subfolder, sequence)
        steering_path = os.path.join(sequence_path,
                                     'vehicle_data/steering.txt')
        throttle_path = os.path.join(sequence_path,
                                     'vehicle_data/throttle.txt')
        brake_path = os.path.join(sequence_path,
                                  'vehicle_data/brake.txt')
        steering_raw = pd.read_csv(steering_path,
                                   header=None)[0].values.tolist()
        throttle_raw = pd.read_csv(throttle_path,
                                   header=None)[0].values.tolist()
        brake_raw = pd.read_csv(brake_path,
                                header=None)[0].values.tolist()

        # take one sample every 40 samples,
        for number, item in enumerate(steering_raw):
            if number % 40 == 0:
                steering_valid.append(item)

        for number, item in enumerate(throttle_raw):
            if number % 40 == 0:
                throttle_valid.append(item)

        for number, item in enumerate(brake_raw):
            if number % 40 == 0:
                brake_valid.append(item)

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
bin_s = np.arange(-0.45, 0.475, 0.025).tolist()
bin_t = np.arange(0, 0.825, 0.025).tolist()
bin_b = np.arange(0, 1.025, 0.025).tolist()
# steering
plt.clf()

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('CARLA: steering distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(steering_train, bins=bin_s, color='g')
plt.xlabel('Steering command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('CARLA: steering distribution of validation data',
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
ax1.set_title('CARLA: throttle distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(throttle_train, bins=bin_t, color='g')
plt.xlabel('Throttle command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('CARLA: throttle distribution of validation data',
              fontsize=11,
              fontweight='bold')
ax2.hist(throttle_valid, bins=bin_t, color='brown')
plt.xlabel('Throttle command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
fig.savefig(SDIR + "/throttle_distribution.pdf")

# brake
plt.clf()

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('CARLA: brake distribution of training data',
              fontsize=11,
              fontweight='bold')
ax1.hist(brake_train, bins=bin_b, color='g')
plt.xlabel('Brake command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('CARLA: brake distribution of validation data',
              fontsize=11,
              fontweight='bold')
ax2.hist(brake_valid, bins=bin_b, color='brown')
plt.xlabel('Brake command', fontsize=11)
plt.ylabel('Number', fontsize=11)
plt.grid(ls='--')
fig.savefig(SDIR + "/brake_distribution.pdf")

F = SDIR + "mean&var.txt"
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
