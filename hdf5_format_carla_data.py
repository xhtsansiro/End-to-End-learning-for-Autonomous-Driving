"""Convert format of image to hdf5."""

# /sequence :
#     ->Sequence0:
#        -> pics,           -> actions:
#           -> pic1,               -> act1,
#           -> pic2 ...            -> act2 ...

import os
import time
import argparse
import h5py
import numpy as np
from PIL import Image
import pandas as pd
from utils import make_dirs


parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter)

# mode ---------------------------------------------------------------------
parser.add_argument("--type", type=str, default='train')

# Parse arguments ----------------------------------------------------------
args = parser.parse_args()

if args.type == 'train':
    DATA_NAME = "/home/ubuntu/repos/EventScape_hdf5/train_data"
    DIR_HDF5 = "/home/ubuntu/repos/EventScape_hdf5/train_sequence_whole"

elif args.type == 'valid':
    DATA_NAME = "/home/ubuntu/repos/EventScape_hdf5/valid_data"
    DIR_HDF5 = "/home/ubuntu/repos/EventScape_hdf5/valid_sequence_whole"
else:
    DATA_NAME = "/home/ubuntu/repos/EventScape_hdf5/test_data"
    DIR_HDF5 = "/home/ubuntu/repos/EventScape_hdf5/test_sequence_whole"

PIC_FOLDER = "rgb/data"

start = time.time()

data_town = os.listdir(DATA_NAME)
data_town = list(filter(lambda a: '.DS_Store' not in a, data_town))
# data_town = list(filter(lambda a: 'sequence' not in a, data_town))
print(data_town)
SEQUENCE_COUNTER = 0

# create the directory
DIR = DIR_HDF5+"/"
make_dirs(DIR)

# create the hdf5 group
NAME_H5PY = DIR_HDF5 + "/All_Sequence"
# where to store the hdf5 file, the name is all_sequence
hf = h5py.File(NAME_H5PY, 'a')
# create this hdf5 group file.


for i in data_town:  # i is the town name
    subfolder = os.path.join(DATA_NAME, i)
    # e.g /Volumes/TOSHIBA EXT/Town01-03_train/Town01/
    sequence = os.listdir(subfolder)
    sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))
    # use filter to remove .DS_Store file in the list,
    #  print(sequence) #e.g. sequence_0, sequence_1, ...., sequence_n
    for j in sequence:  # in each sequences, j is the sequence name

        rgb_folder = os.path.join(subfolder, j, PIC_FOLDER)
        # where all pics are saved
        grp = hf.create_group('sequence'+str(SEQUENCE_COUNTER))
        # create hdf5 group
        SEQUENCE_COUNTER += 1

        # read the steering,throttle, brake data of each sequence,
        steering_path = os.path.join(
            subfolder, j, "vehicle_data/steering.txt")
        throttle_path = os.path.join(
            subfolder, j, "vehicle_data/throttle.txt")
        brake_path = os.path.join(subfolder, j, "vehicle_data/brake.txt")
        steering_raw = pd.read_csv(steering_path, header=None)[0].\
            values.tolist()
        throttle_raw = pd.read_csv(throttle_path, header=None)[0].\
            values.tolist()
        brake_raw = pd.read_csv(brake_path, header=None)[0].values.tolist()

        for root, _, pic_names in sorted(os.walk(rgb_folder)):
            C = 0
            A = 0
            subgrp1 = grp.create_group("pics")
            subgrp2 = grp.create_group("actions")
            for pic_name in sorted(pic_names):
                if "timestamps" not in pic_name and \
                        ".DS_Store" not in pic_name:

                    # images
                    img_path = os.path.join(rgb_folder, pic_name)
                    img_np = np.array(Image.open(img_path), dtype=np.uint8)
                    # actions
                    actions = [steering_raw[A], throttle_raw[A], brake_raw[A]]
                    # a list of action pair
                    actions = np.array(actions, dtype=np.float32)
                    # change the list to the array

                    dset1 = subgrp1.create_dataset(
                        "pic" + str(C), data=img_np)
                    dset2 = subgrp2.create_dataset(
                        "act" + str(C), data=actions)
                    C += 1
                    A += 40  # take one action each 40 records

hf.close()  # after iterating all the files and the

print(f"the format change totally takes {time.time() - start}")
F = DIR_HDF5 + "/summary.txt"  # output the information of transformation
SUMM = "totally " + str(SEQUENCE_COUNTER) + \
       " sequences are changed to hdf5 format. And it takes " \
       + str(time.time() - start) + "seconds."
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(SUMM)
