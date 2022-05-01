"""This script is used to train CNN+NCP."""

# train NvidiaCNN+NCP, AlexNet+NCP, ResNet+NCP
# use Zurich event Data to train NCP，aiming to find the suitable CNN head
import sys
import argparse
import time
import numpy as np
import torch
from torchvision import transforms
import h5py
from wiring import NCPWiring
from nets.cnn_head import ConvolutionHead_Nvidia, \
    ConvolutionHead_ResNet, ConvolutionHead_AlexNet
from nets.ltc_cell import LTCCell
from nets.models_all import NCP_Model
from dataset_all import MarkovProcessRNNHdf5, MarkovProcessRNNPng
from early_stopping import EarlyStopping
from utils import make_dirs, save_result, __crop, epoch_policy, \
    evaluate_on_single_sequence

start_time = time.time()
parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter)

# mode -------------------------------------------------------------------
parser.add_argument("--cluster", type=str, default='no')
parser.add_argument("--name", type=str, default='Training_Result')
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--sequence", type=int, default=16)
parser.add_argument("--hdf5", type=str, default='no')
# use hdf5 file as input, default is no.
parser.add_argument("--cnn_head", type=str, default='nvidia')
# nvidia / alexnet / resnet
parser.add_argument("--seed", type=int, default=100)  # default seed,

# Parse arguments --------------------------------------------------------
args = parser.parse_args()

if args.cluster == 'yes':  # when using GPU
    DATA_NAME_TRAIN = "/home/ubuntu/repos/EventScape_hdf5/train_data"
    DATA_NAME_VALID = "/home/ubuntu/repos/EventScape_hdf5/valid_data"

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

# hyperparameters

BATCH_SIZE = args.batch
seq_length = args.sequence
EARLY_LENGTH = 5
N_EPOCH = args.epoch
# specify the random seed
seed_value = args.seed
torch.manual_seed(seed_value)  # for CPU
torch.cuda.manual_seed(seed_value)  # for GPU
np.random.seed(seed_value)

# s_dim, a_dim of the pic.
S_DIM = (3, 66, 200)
A_DIM = 3

# prepare save directories
sdir = SAVE_DIR + "/"
make_dirs(sdir)

# define the transform
transform = None if isinstance(S_DIM, int) else \
    transforms.Compose([transforms.ToPILImage(),
                        transforms.Lambda(
                            lambda img:__crop(img, (10, 80), (500, 176))
                        ),
                        transforms.Resize((66, 200)),
                        transforms.ToTensor()]
                       )

# set expert dataset

if args.hdf5 == 'no':   # use png files
    dataset_train = MarkovProcessRNNPng(DATA_NAME_TRAIN,
                                        time_step=seq_length,
                                        transform=transform,
                                        mode='train')
    dataset_valid = MarkovProcessRNNPng(DATA_NAME_VALID,
                                        time_step=seq_length,
                                        transform=transform,
                                        mode='eval')

elif args.hdf5 == 'yes':  # use hdf5 files
    hf5_train = h5py.File(HDF5_TRAIN, 'r')
    # read the hdf5 group of train data
    hf5_valid = h5py.File(HDF5_VALID, 'r')
    # read the hdf5 group of valid data

    dataset_train = MarkovProcessRNNHdf5(hf5_train,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='train')
    dataset_valid = MarkovProcessRNNHdf5(hf5_valid,
                                         time_step=seq_length,
                                         transform=transform,
                                         mode='eval')


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

######################################################
# generate an object of Class NCP_Wiring, Class ConvolutionHead, Class LTCCell
if args.cnn_head == 'nvidia':
    cnn_head = ConvolutionHead_Nvidia(S_DIM,
                                      seq_length,
                                      num_filters=32,
                                      features_per_filter=4)
elif args.cnn_head == 'resnet':
    cnn_head = ConvolutionHead_ResNet(S_DIM,
                                      seq_length,
                                      num_filters=32,
                                      features_per_filter=4)
elif args.cnn_head == 'alexnet':
    cnn_head = ConvolutionHead_AlexNet(S_DIM,
                                       seq_length,
                                       num_filters=32,
                                       features_per_filter=4)
else:
    print('Invalid cnn head setting: ', args.cnn_head)
    sys.exit()
# input shape for NCP, which is also the sensory neuron number
input_shape = (1, cnn_head.num_filters * cnn_head.features_per_filter)

# initialize the wiring, and connect between layers
wiring = NCPWiring(inter_neurons=64, command_neurons=32,
                   motor_neurons=3, sensory_fanout=48,
                   inter_fanout=24, recurrent_command=24, motor_fanin=16)

wiring.build(input_shape)

ltc_cell = LTCCell(wiring=wiring, time_interval=0.04)

# create an object which output the output of the NCP
policy = NCP_Model(ltc_cell=ltc_cell, conv_head=cnn_head)
print(policy)  # show the network details in terminal
if str(policy.device) == "cuda":    # feed models to GPU
    print("there is GPU")
    policy.to(policy.device)

print(f"the total params: {cnn_head.count_params()}"
      f"for CNN, {policy.count_params()} for NCP.")

stopper = EarlyStopping(length=EARLY_LENGTH)  # to avoid overfitting

# prepare buffers to store results
train_loss_policy_o = []
valid_loss_policy_o = []

# optimize policy by the expert dataset
print("Start learning policy!")
for n_epoch in range(1, N_EPOCH + 1):
    # seed must always changing, otherwise it is the same split every time

    l_origin = epoch_policy(train_loader, policy, n_epoch, "train", 'NCP')
    train_loss_policy_o.append(l_origin)

    with torch.no_grad():
        l_origin = evaluate_on_single_sequence(
            dataset_valid.sequence_number,
            dataset_valid,
            policy,
            n_epoch,
            "valid"
        )
        valid_loss_policy_o.append(l_origin)

    # early stopping
    if stopper(valid_loss_policy_o[-1]):
        # use origin loss for the early stopping
        print("Early Stopping to avoid overfitting!")
        break

# save trained model
policy.release(sdir)

# csv and plot
save_result(sdir,
            "loss_policy_origin",
            {"train": train_loss_policy_o, "valid": valid_loss_policy_o})

# close the hdf5 file.
hf5_train.close()
hf5_valid.close()

# output the parameters and settings of the NN:

dict1 = {'batch_size': BATCH_SIZE,
         'time_sequence': seq_length,
         'total_params of NCP': policy.count_params(),
         'total_params of CNN head': cnn_head.count_params()}
dict2 = {'network_info': policy.nn_structure()}
dict3 = wiring.get_config()

dict_whole = {'Basic information': dict1,
              'Network info': dict2,
              'NCP details': dict3}

path = sdir + "/settings.pth"
torch.save(dict_whole, path)

# calculate the total execution time of the code
end_time = time.time()
execution_time = end_time - start_time
hours = execution_time//3600
mins = (execution_time % 3600) // 60
seconds = (execution_time % 3600) % 60
print(f"The execution time "
      f"is {hours}h {mins}m {seconds}s")

# output the time duration of the training.
F = sdir + "/summary.txt"  # output the information of transformation
SUMM = " Totally" + str(n_epoch) + "training are done, it takes " + \
       str(hours) + "h " + str(mins) + "min " + str(seconds) + "s"

with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(SUMM)
