"""Train all the models.

Use CARLA/LGSVL Data to train NCP, CTGRU, LSTM, RNN, CNN.
"""

import sys
import argparse
import time
import numpy as np
import torch
from torchvision import transforms
import h5py
from wiring import NCPWiring
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.ltc_cell import LTCCell
from nets.models_all import Convolution_Model, GRU_Model, LSTM_Model, \
    CTGRU_Model, NCP_Model
from dataset_all import MarkovProcessRNNHdf5, MarkovProcessCNNHdf5
from early_stopping import EarlyStopping
from utils import make_dirs, save_result, __crop, epoch_policy, \
    evaluate_on_single_sequence

start_time = time.time()
parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)

# mode ------------------------------------------------------------------
parser.add_argument("--cluster", type=str, default='no')
parser.add_argument("--data", type=str, default='CARLA')
# CARLA or LGSVL

parser.add_argument("--network", type=str, default='CNN')
# CNN, GRU, LSTM, CTGRU, NCP
parser.add_argument("--name", type=str, default='Training_Result')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--sequence", type=int, default=16)  # only for RNN
parser.add_argument("--hidden", type=int, default=64)   # hidden units
# 64 for throttle, brake, steering, 16 for only steering.
parser.add_argument("--output", type=int, default=3)   # output predictions
# 3 for throttle, brake, steering, 1 for only steering.
parser.add_argument("--seed", type=int, default=100)  # default seed,


# Parse arguments -------------------------------------------------------
args = parser.parse_args()

if args.cluster == 'yes':  # when using GPU
    if args.data == 'LGSVL':
        HDF5_TRAIN = "/home/ubuntu/repos/LGSVL_hdf5/" \
                     "train_data/All_Sequence"
        HDF5_VALID = "/home/ubuntu/repos/LGSVL_hdf5/" \
                     "valid_data/All_Sequence"
    else:
        HDF5_TRAIN = "/home/ubuntu/repos/EventScape_hdf5/" \
                      "train_sequence_whole/All_Sequence"
        HDF5_VALID = "/home/ubuntu/repos/EventScape_hdf5/" \
                     "valid_sequence_whole/All_Sequence"

    SAVE_DIR = "/home/ubuntu/repos/results_haotian/" + args.name + "/"
elif args.cluster == 'no':
    DATA_NAME_TRAIN = "/Volumes/TOSHIBA EXT/train_data"
    DATA_NAME_VALID = "/Volumes/TOSHIBA EXT/validation_data"
    # for debug, train only on one sequence

    SAVE_DIR = "./result/" + args.name + "/"
else:
    print('Invalid cluster setting: ', args.cluster)
    sys.exit()

BATCH_SIZE = args.batch
seq_length = args.sequence
EARLY_LENGTH = 5     # change from 5 to 45
N_EPOCH = args.epoch  # for debug, in order to accelerate
METHOD = args.name
Network = args.network
HIDDEN_UNITS = args.hidden
OUTPUT = args.output
seed_value = args.seed
torch.manual_seed(seed_value)  # for CPU
torch.cuda.manual_seed(seed_value)  # for GPU
np.random.seed(seed_value)


# s_dim, a_dim of the pic.
s_dim = (3, 66, 200)
a_dim = args.output

hf5_train = h5py.File(HDF5_TRAIN, 'r')  # read the hdf5 group of train data
hf5_valid = h5py.File(HDF5_VALID, 'r')  # read the hdf5 group of valid data

# prepare save directories
SDIR = SAVE_DIR + "/"
make_dirs(SDIR)

if args.data == 'LGSVL':
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Lambda(lambda img:__crop(img, (100, 80), (528, 176))),
         transforms.Resize((66, 200)),
         transforms.ToTensor()]
    )
else:   # for CARLA
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Lambda(lambda img: __crop(img, (10, 80), (500, 176))),
         transforms.Resize((66, 200)),
         transforms.ToTensor()]
    )

if Network == 'CNN':
    dataset_train = MarkovProcessCNNHdf5(hf5_train,
                                         output=OUTPUT,
                                         transform=transform,
                                         mode='train')
    dataset_valid = MarkovProcessCNNHdf5(hf5_valid,
                                         output=OUTPUT,
                                         transform=transform,
                                         mode='eval')

elif Network in ["GRU", "LSTM", "CTGRU", "NCP"]:
    dataset_train = MarkovProcessRNNHdf5(hf5_train,
                                         output=OUTPUT,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='train')
    dataset_valid = MarkovProcessRNNHdf5(hf5_valid,
                                         output=OUTPUT,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='eval')
else:
    print('unknown network type: ', Network)
    sys.exit()

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=16,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=16,
                                           pin_memory=True)
POLICY = None

if Network == 'CNN':
    POLICY = Convolution_Model(s_dim, a_dim)   # initialize the CNN model
elif Network == 'GRU':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = GRU_Model(cnn_head,
                       time_step=seq_length,
                       input_size=32*4,
                       hidden_size=HIDDEN_UNITS,
                       output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands

elif Network == 'LSTM':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = LSTM_Model(cnn_head,
                        time_step=seq_length,
                        input_size=32*4,
                        hidden_size=HIDDEN_UNITS,
                        output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands

elif Network == 'CTGRU':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32,
                                      features_per_filter=4)  # CNN before RNN
    POLICY = CTGRU_Model(num_units=HIDDEN_UNITS,
                         conv_head=cnn_head,
                         output=OUTPUT)
    # 1 for predicting only steering, 3 for all commands
    # time interval:0.04s for training, 0.2s for test.
    # in line 421 of models_all.py

elif Network == 'NCP':
    cnn_head = ConvolutionHead_Nvidia(s_dim, seq_length,
                                      num_filters=32, features_per_filter=4)
    input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)
    # This is for predicting all the actions.
    if OUTPUT == 3:   # predict steering, throttle, brake
        wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                           motor_neurons=3, sensory_fanout=48,
                           inter_fanout=24, recurrent_command=24,
                           motor_fanin=16)
    else:  # predict only steering
        wiring = NCPWiring(inter_neurons=22, command_neurons=12,
                           motor_neurons=1, sensory_fanout=16,
                           inter_fanout=8, recurrent_command=8,
                           motor_fanin=6)
    wiring.build(input_shape)
    # time interval between 2 pics is 0.04s.
    ltc_cell = LTCCell(wiring=wiring, time_interval=0.04)
    POLICY = NCP_Model(ltc_cell=ltc_cell, conv_head=cnn_head)


if str(POLICY.device) == "cuda":    # feed models to GPU
    print("there is GPU")
    POLICY.to(POLICY.device)

print(POLICY)
print(f"the total params of Network {POLICY.count_params()}")
# print(f"the params of cnn head {POLICY.conv_head.count_params()}")

stopper = EarlyStopping(length=EARLY_LENGTH)  # to avoid over fitting,

# prepare buffers to store results
train_loss_policy_o = []
valid_loss_policy_o = []

print("Start learning policy!")
for n_epoch in range(1, N_EPOCH + 1):

    l_origin = epoch_policy(train_loader, POLICY, n_epoch, "train", Network)
    train_loss_policy_o.append(l_origin)
    with torch.no_grad():
        if Network == 'CNN':
            valid_loss_policy_o.append(epoch_policy(
                valid_loader,
                POLICY,
                n_epoch,
                "valid",
                Network)
            )
        else:
            l_origin = evaluate_on_single_sequence(
                dataset_valid.sequence_number,
                dataset_valid,
                POLICY,
                n_epoch,
                "valid")
            valid_loss_policy_o.append(l_origin)

    # policy.scheduler.step()  # update the lr rate after every set epoch
    # early stopping
    if stopper(valid_loss_policy_o[-1]):  # origin loss
        print("Early Stopping to avoid Overfitting!")
        break

# save trained model
POLICY.release(SDIR)

# close the hdf5 file.
hf5_train.close()
hf5_valid.close()

# csv and plot
save_result(SDIR,
            "loss_policy_origin",
            {"train": train_loss_policy_o, "valid": valid_loss_policy_o}
            )

# output the parameters and settings of the NN，
# save the params, varies from network to network:
# the structure of the network, fetch it before sending to GPU
dict_layer = POLICY.nn_structure()
if Network == 'CNN':
    dict_params = {'batch_size': BATCH_SIZE,
                   'total_params of Network': POLICY.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'GRU':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'LSTM':
    dict_params = {'batch_size': BATCH_SIZE,
                   'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'CTGRU':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params}
elif Network == 'NCP':
    dict_params = {'batch_size': BATCH_SIZE, 'time_sequence': seq_length,
                   'total_params of Network': POLICY.count_params(),
                   'params of CNN_head': POLICY.conv_head.count_params()}
    dict_wiring = wiring.get_config()  # ltc_cell._wiring.get_config()
    dict_whole = {'layer information': dict_layer,
                  'param information': dict_params,
                  'NCP Wiring': dict_wiring}

path = SDIR + "/network_settings.pth"
torch.save(dict_whole, path)

# calculate the total execution time of the code
end_time = time.time()
execution_time = end_time - start_time
hours = execution_time//3600
mins = (execution_time % 3600) // 60
seconds = (execution_time % 3600) % 60
print(f"The execution time is {hours}h {mins}m {seconds}s")

# output the time duration of the training.
F = SDIR + "/summary.txt"  # output the information of transformation
SUMM = " Totally" + str(n_epoch) + "training are done, it takes " \
       + str(hours) + "h " + str(mins) + "min " + str(seconds) + "s"
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(SUMM)
