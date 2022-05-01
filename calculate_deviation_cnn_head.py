"""Calculate average absolute deviation of CNN+NC.

This file is used to assess the absolute deviation
between predicted actions and ground truth,
Comparison among CNN heads: NvidiaCNN, AlexNet, ResNet.
comment line 641,642 of models_all,
uncomment line 644-646 of models_all
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from utils import make_dirs, __crop
from wiring import NCPWiring
from nets.models_all import NCP_Model
from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_AlexNet, \
    ConvolutionHead_ResNet
from nets.ltc_cell import LTCCell


parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--cluster", type=str, default='no')
args = parser.parse_args()

if args.cluster == 'yes':  # use GPU, in Laufwerk
    # folder to save the result
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "06_CNN_head_comparison/04_Comparison/"
    # folder where the sequences are saved .../test_data/Town05
    TEST_DATA = "/home/ubuntu/repos/EventScape/valid_data/Town05"
    # the number at the end of the name represents the used seed.
    # NvidiaCNN+NCP
    NCP_N_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "01_NvidiaCNN+NCP/seed_100/"
    NCP_N_150 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "01_NvidiaCNN+NCP/seed_150/"
    NCP_N_200 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "01_NvidiaCNN+NCP/seed_200/"

    # AlexNet+NCP
    NCP_A_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "02_AlexNet+NCP/seed_100/"
    NCP_A_150 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "02_AlexNet+NCP/seed_150/"
    NCP_A_200 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "02_AlexNet+NCP/seed_200/"

    # ResNet+NCP
    NCP_R_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "03_ResNet+NCP/seed_100/"
    NCP_R_150 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "03_ResNet+NCP/seed_150/"
    NCP_R_200 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
                "06_CNN_head_comparison/05_Trained_model/" \
                "03_ResNet+NCP/seed_200/"

elif args.cluster == 'no':  # in my own PC
    SDIR = "xxxx"     # folder to save the result
    TEST_DATA = "xxxx"   # folder where the sequences are saved
    # prepare save directories, depends on where models are saved locally.
else:
    print('Invalid cluster setting: ', args.cluster)
    sys.exit()

path_nvidia = [NCP_N_100, NCP_N_150, NCP_N_200]
path_alex = [NCP_A_100, NCP_A_150, NCP_A_200]
path_res = [NCP_R_100, NCP_R_150, NCP_R_200]

PIC_FOLDER = "rgb/data"
STEERING_PATH = "vehicle_data/steering.txt"
THROTTLE_PATH = "vehicle_data/throttle.txt"
BRAKE_PATH = "vehicle_data/brake.txt"

dev_nvi_s, dev_nvi_t, dev_nvi_b = [], [], []
# deviation of Nvidia, steering, throttle, brake
dev_alex_s, dev_alex_t, dev_alex_b = [], [], []
# deviation of AlexNet: steering, throttle, brake
dev_res_s, dev_res_t, dev_res_b = [], [], []
# deviation of ResNet: steering, throttle, brake

S_DIM = (3, 66, 200)
A_DIM = 3
SEQ_LENGTH = 1  # takes each image sequentially.
transform = None if isinstance(S_DIM, int) else \
    transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Lambda(lambda img:__crop(img, (10, 80), (500, 176))),
         transforms.Resize((66, 200)),
         transforms.ToTensor()]
    )

# ground truth of steering,throttle, brake.
s_truth, t_truth, b_truth = [], [], []

# iterate all models, 0: seed 100, 1: seed 150, 2: seed 200
for index in range(3):
    # declare Nvidia CNN head
    cnn_head_n = ConvolutionHead_Nvidia(
        S_DIM,
        SEQ_LENGTH,
        num_filters=32,
        features_per_filter=4
    )
    # declare AlexNet CNN head
    cnn_head_a = ConvolutionHead_AlexNet(
        S_DIM,
        SEQ_LENGTH,
        num_filters=32,
        features_per_filter=4
    )
    # declare ResNet CNN head
    cnn_head_r = ConvolutionHead_ResNet(
        S_DIM,
        SEQ_LENGTH,
        num_filters=32,
        features_per_filter=4
    )

    # declare the object of NCP
    input_shape = (1, 32*4)  # the same for every cnn head.
    # cnn_head_a.num_filters * cnn_head.features_per_filter
    # for nvidia+NCP
    wiring_n = NCPWiring(inter_neurons=64, command_neurons=32, motor_neurons=3,
                         sensory_fanout=48, inter_fanout=24,
                         recurrent_command=24, motor_fanin=16)
    # for AlexNet+NCP
    wiring_a = NCPWiring(inter_neurons=64, command_neurons=32, motor_neurons=3,
                         sensory_fanout=48, inter_fanout=24,
                         recurrent_command=24, motor_fanin=16)
    # for ResNet+NCP
    wiring_r = NCPWiring(inter_neurons=64, command_neurons=32, motor_neurons=3,
                         sensory_fanout=48, inter_fanout=24,
                         recurrent_command=24, motor_fanin=16)

    wiring_n.build(input_shape)
    wiring_a.build(input_shape)
    wiring_r.build(input_shape)

    # time interval between 2 consecutive pics is 0.04s.
    ltc_cell_n = LTCCell(wiring=wiring_n, time_interval=0.04)
    ltc_cell_a = LTCCell(wiring=wiring_a, time_interval=0.04)
    ltc_cell_r = LTCCell(wiring=wiring_r, time_interval=0.04)

    policy_n = NCP_Model(ltc_cell=ltc_cell_n, conv_head=cnn_head_n)
    policy_a = NCP_Model(ltc_cell=ltc_cell_a, conv_head=cnn_head_a)
    policy_r = NCP_Model(ltc_cell=ltc_cell_r, conv_head=cnn_head_r)
    # load trained model
    policy_n.load(path_nvidia[index])
    policy_n.eval()  # in evaluation mode

    policy_a.load(path_alex[index])
    policy_a.eval()  # in evaluation mode

    policy_r.load(path_res[index])
    policy_r.eval()  # in evaluation mode

    s_p_N, s_p_A, s_p_R = [], [], []  # steering prediction Nvidia/Alex/ResNet

    t_p_N, t_p_A, t_p_R = [], [], []  # throttle prediction Nvidia/Alex/ResNet

    b_p_N, b_p_A, b_p_R = [], [], []  # brake prediction Nvidia/Alex/ResNet

    sequence = os.listdir(TEST_DATA)
    # how many sequences need to be visualized
    sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))
    # ensure every trial it has the same sequnce.
    sequence = sorted(sequence)

# plot for only one sequence, iterate over the whole data,
    for j in sequence:

        PATH = TEST_DATA + "/" + j
        img_path = os.path.join(PATH, PIC_FOLDER)

        list_imgs = sorted(os.listdir(img_path))
        # get all the name of the images
        hidden_state_N, hidden_state_A, hidden_state_R = None, None, None

    # sequence = list(filter(lambda a: '.DS_Store' not in a, list_imgs))

        for pic_name in list_imgs:
            if "timestamps" not in pic_name and \
                    ".DS_Store" not in pic_name and "._" not in pic_name:
                pic_path = os.path.join(img_path, pic_name)
                # the rgb_image path
                states = np.asarray(Image.open(pic_path), dtype=np.uint8)
                states_CNN = transform(states)  # this is for CNN
                # input of CNN_head, should be B,T,C,H,W,
                # needs to broadcast for use.
                states_NCP = torch.unsqueeze(states_CNN, 0)
                states_NCP = torch.unsqueeze(states_NCP, 0)

                # prediction of Nvidia CNN + NCP
                act_pairs_N, hidden_state_N = \
                    policy_n.evaluate_on_single_sequence(
                        states_NCP,
                        hidden_state=hidden_state_N
                    )
                # print(act_pairs_N)

                s_p_N.append(float(act_pairs_N[0][0][0]))
                # the output has size (BS,action_pairs)
                t_p_N.append(float(act_pairs_N[0][0][1]))
                b_p_N.append(float(act_pairs_N[0][0][2]))

                # prediction of AlexNet + NCP
                act_pairs_A, hidden_state_A = \
                    policy_a.evaluate_on_single_sequence(
                        states_NCP,
                        hidden_state=hidden_state_A
                    )
                s_p_A.append(float(act_pairs_A[0][0][0]))
                # the output has size (BS,action_pairs)
                t_p_A.append(float(act_pairs_A[0][0][1]))
                b_p_A.append(float(act_pairs_A[0][0][2]))

                # prediction of ResNet + NCP
                act_pairs_R, hidden_state_R = \
                    policy_r.evaluate_on_single_sequence(
                        states_NCP,
                        hidden_state=hidden_state_R
                    )
                # print(act_pairs_N)
                s_p_R.append(float(act_pairs_R[0][0][0]))
                # the output has size (BS,action_pairs)
                t_p_R.append(float(act_pairs_R[0][0][1]))
                b_p_R.append(float(act_pairs_R[0][0][2]))

        # take the ground truth
        if index == 0:  # only fetch it at the first iteration.
            steering_path_full = os.path.join(PATH, STEERING_PATH)
            throttle_path_full = os.path.join(PATH, THROTTLE_PATH)
            brake_path_full = os.path.join(PATH, BRAKE_PATH)

            # read data of steering, throttle, brake
            steering_raw = pd.read_csv(steering_path_full,
                                       header=None)[0].values.tolist()

            throttle_raw = pd.read_csv(throttle_path_full,
                                       header=None)[0].values.tolist()

            brake_raw = pd.read_csv(brake_path_full,
                                    header=None)[0].values.tolist()

            print(len(brake_raw))

            # take one sample every 40 samples,
            for number, item in enumerate(steering_raw):
                if number % 40 == 0:
                    s_truth.append(item)

            for number, item in enumerate(throttle_raw):
                if number % 40 == 0:
                    t_truth.append(item)

            for number, item in enumerate(brake_raw):
                if number % 40 == 0:
                    b_truth.append(item)

    print(len(s_truth))
    print(len(s_p_N))

    # len(s_truth) == len(s_p_CNN) == len(s_p_NCP_N) == len(s_p_NCP_A)
    # == len(s_p_NCP_R)
    assert \
        len(s_truth) == len(s_p_N) == len(s_p_A) == len(s_p_R), \
        "check number of steering commands"
    assert \
        len(t_truth) == len(t_p_N) == len(t_p_A) == len(t_p_R), \
        "check number of throttle commands"
    assert \
        len(b_truth) == len(b_p_N) == len(b_p_A) == len(b_p_R), \
        "check number of brake commands"
    assert len(s_truth) == len(t_truth) == len(b_truth), "check csv file"

    # absolute deviation of Nvidia+NCP
    dev_nvi_s.append(np.absolute(np.array(s_truth) - np.array(s_p_N)).tolist())
    dev_nvi_t.append(np.absolute(np.array(t_truth) - np.array(t_p_N)).tolist())
    dev_nvi_b.append(np.absolute(np.array(b_truth) - np.array(b_p_N)).tolist())
    # absolute deviation of Alex+NCP
    dev_alex_s.append(
        np.absolute(np.array(s_truth) - np.array(s_p_A)).tolist()
    )
    dev_alex_t.append(
        np.absolute(np.array(t_truth) - np.array(t_p_A)).tolist()
    )
    dev_alex_b.append(
        np.absolute(np.array(b_truth) - np.array(b_p_A)).tolist()
    )
    # absolute deviation of ResNet+NCP
    dev_res_s.append(np.absolute(np.array(s_truth) - np.array(s_p_R)).tolist())
    dev_res_t.append(np.absolute(np.array(t_truth) - np.array(t_p_R)).tolist())
    dev_res_b.append(np.absolute(np.array(b_truth) - np.array(b_p_R)).tolist())

# after iteration, the dev_nvi_s is [[dev_seed100],...,[dev_seed200]].
# change to array, then can calculate its mean and save as a list.
print(len(dev_nvi_s))  # the length of the seeds
print(len(dev_nvi_s[0]))  # the length of how many steerings,
dev_nvi_s = (np.array(dev_nvi_s).mean(axis=0)).tolist()
dev_nvi_t = (np.array(dev_nvi_t).mean(axis=0)).tolist()
dev_nvi_b = (np.array(dev_nvi_b).mean(axis=0)).tolist()

dev_alex_s = (np.array(dev_alex_s).mean(axis=0)).tolist()
dev_alex_t = (np.array(dev_alex_t).mean(axis=0)).tolist()
dev_alex_b = (np.array(dev_alex_b).mean(axis=0)).tolist()

dev_res_s = (np.array(dev_res_s).mean(axis=0)).tolist()
dev_res_t = (np.array(dev_res_t).mean(axis=0)).tolist()
dev_res_b = (np.array(dev_res_b).mean(axis=0)).tolist()

make_dirs(SDIR + "absolute_deviation_distribution/")
# sequence idx needs change

# save the list of deviation in the csv file.
data = {'deviation of steering for NvidiaCNN+NCP': dev_nvi_s,
        'deviation of steering for AlexNet+NCP': dev_alex_s,
        'deviation of steering for ResNet+NCP': dev_res_s,
        'deviation of throttle for NvidiaCNN+NCP': dev_nvi_t,
        'deviation of throttle for AlexNet+NCP': dev_alex_t,
        'deviation of throttle for ResNet+NCP': dev_res_t,
        'deviation of brake for NvidiaCNN+NCP': dev_nvi_b,
        'deviation of brake for AlexNet+NCP': dev_alex_b,
        'deviation of brake for ResNet+NCP': dev_res_b,
        }
pd.DataFrame(data).to_csv(
    SDIR + "absolute_deviation_distribution/" + "deviation.csv")
