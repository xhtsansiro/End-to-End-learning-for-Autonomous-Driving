"""Convert format of image to hdf5."""

# /sequence :
#     ->Sequence0:
#        -> pics,           -> actions:
#           -> pic1,               -> act1,
#           -> pic2 ...            -> act2 ...

import os
import time
import argparse
import sys
import h5py
import numpy as np
from PIL import Image
import pandas as pd
from utils import make_dirs

parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)

# mode ---------------------------------------------------------------------
parser.add_argument("--type", type=str, default='train')
# train or valid

# Parse arguments ----------------------------------------------------------
args = parser.parse_args()

if args.type == 'train':
    DATA_NAME = "/home/ubuntu/repos/LGSVL/train_data"  # valid_data
    DIR_HDF5 = "/home/ubuntu/repos/LGSVL_hdf5/train_data"    # valid_data
elif args.type == 'no':
    DATA_NAME = "/home/ubuntu/repos/LGSVL/valid_data"  # valid_data
    DIR_HDF5 = "/home/ubuntu/repos/LGSVL_hdf5/valid_data"  # valid_data
else:
    print('Invalid type: ', args.type)
    sys.exit()

PIC_FOLDER = "img"

start = time.time()

data_town = os.listdir(DATA_NAME)
data_town = sorted(list(filter(lambda a: '.DS_Store' not in a, data_town)))
data_town = list(filter(lambda a: '._Town' not in a, data_town))
print(data_town)
SEQUENCE_COUNTER = 0

# create the directory
DIR = DIR_HDF5+"/"
make_dirs(DIR)

# create the hdf5 group
NAME_H5PY = DIR_HDF5 + "/All_Sequence"
# where to store the hdf5 file, the name is all_sequence
hf = h5py.File(NAME_H5PY, 'a')  # create this hdf5 group file.

for i in data_town:  # i is the town name
    subfolder = os.path.join(DATA_NAME, i)  # e.g .../Town03
    print(subfolder)
    sequence = sorted(os.listdir(subfolder))
    # get all the sequences path in this town, sequence1/2.../6
    sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))
    # use filter to remove .DS_Store file in the list,
    print(sequence)  # e.g. sequence_0, sequence_1, ...., sequence_n
    for j in sequence:  # in each sequences, j is the sequence name
        # where all pics are saved
        rgb_folder = os.path.join(subfolder, j, PIC_FOLDER)
        #  name_h5py = dir_hdf5 + "/" + "sequence" + str(sequence_counter)
        # create hdf5 group
        grp = hf.create_group('sequence'+str(SEQUENCE_COUNTER))
        SEQUENCE_COUNTER += 1

        # read the steering,throttle, brake data of each sequence,
        steering_path = os.path.join(subfolder, j, "action/steering.csv")
        throttle_path = os.path.join(subfolder, j, "action/throttle.csv")
        brake_path = os.path.join(subfolder, j, "action/brake.csv")

        steering_raw = pd.read_csv(steering_path, header=None)[1].\
            values.tolist()
        throttle_raw = pd.read_csv(throttle_path, header=None)[1].\
            values.tolist()
        brake_raw = pd.read_csv(brake_path, header=None)[1].values.tolist()

        # /Volumes/TOSHIBA EXT/Town01-03_train/Town01/sequence_0/rgb/data
        for root, _, pic_names in sorted(os.walk(rgb_folder)):
            C = 0
            A = 0
            subgrp1 = grp.create_group("pics")
            subgrp2 = grp.create_group("actions")
            for pic_name in sorted(pic_names):
                if '._center' not in pic_name and \
                        ".DS_Store" not in pic_name and \
                        '.db' not in pic_name:  # windows has .db file
                    # images
                    img_path = os.path.join(rgb_folder, pic_name)
                    img_np = np.array(Image.open(img_path), dtype=np.uint8)
                    # actions
                    actions = [steering_raw[A], throttle_raw[A], brake_raw[A]]
                    # a list of action pair
                    actions = np.array(actions, dtype=np.float32)
                    # change the list to the array

                    dset1 = subgrp1.create_dataset("pic" + str(C),
                                                   data=img_np)
                    dset2 = subgrp2.create_dataset("act" + str(C),
                                                   data=actions)
                    C += 1
                    A += 1  # take one action
                    # dset = grp.create_dataset(pic_name, data=binary_data_np)
                    # write the data to hdf5 file, the index should be unique
hf.close()  # after iterating all the files and the

print(f"the format change totally takes {time.time() - start}")
F = DIR_HDF5 + "/summary.txt"  # output the information of transformation
SUMM = "totally " + str(SEQUENCE_COUNTER) + \
       " sequences are changed to hdf5 format. And it takes " + \
       str(time.time() - start) + "seconds."
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    # create if not exists
    fil.write(SUMM)
