"""Offline simulation for throttle, brake, steering.

This file is used for sequential plot of commands
for CARLA test dataset.
the prediction value is the average value of each model of one
type neural network.
CNN: 3 models. CNN+LSTM: 3 models. CNN+GRU: 3 models.
CNN+CTGRU: 3 models. CNN+NCP: 3 models.
save the average deviation of each neural network in csv
file for the further analysis.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import make_dirs, __crop
from nets.models_all import Convolution_Model, LSTM_Model, \
    CTGRU_Model, GRU_Model, NCP_Model
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.ltc_cell import LTCCell
from wiring import NCPWiring


parser = argparse.ArgumentParser(
    description="arg parser",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--cluster", type=str, default='no')
args = parser.parse_args()

if args.cluster == 'yes':  # use GPU, in Laufwerk

    TESTDATA = "/home/ubuntu/repos/EventScape/test_data/Town05"
    # folder where the sequences are saved .../test_data/Town05
    SDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "08_Model_comparison/01_Trained_model/"
    RDIR = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
           "08_Model_comparison/03_Comparison/"
    # define where CNN models are saved.
    CNN_S_100 = SDIR + "CNN/all/seed_100/"
    CNN_S_150 = SDIR + "CNN/all/seed_150/"
    CNN_S_200 = SDIR + "CNN/all/seed_200/"
    # define where CNN+LSTM models are saved.
    LSTM_S_100 = SDIR + "CNN+LSTM/all/seed_100/"
    LSTM_S_150 = SDIR + "CNN+LSTM/all/seed_150/"
    LSTM_S_200 = SDIR + "CNN+LSTM/all/seed_200/"
    # define where CNN+GRU models are saved.
    GRU_S_100 = SDIR + "CNN+GRU/all/seed_100/"
    GRU_S_150 = SDIR + "CNN+GRU/all/seed_150/"
    GRU_S_200 = SDIR + "CNN+GRU/all/seed_200/"
    # define where CNN+CTGRU models are saved.
    CTGRU_S_100 = SDIR + "CNN+CTGRU/all/seed_100/"
    CTGRU_S_150 = SDIR + "CNN+CTGRU/all/seed_150/"
    CTGRU_S_200 = SDIR + "CNN+CTGRU/all/seed_200/"
    # define where CNN+NCP models are saved.
    NCP_S_100 = SDIR + "CNN+NCP/all/seed_100/"
    NCP_S_150 = SDIR + "CNN+NCP/all/seed_150/"
    NCP_S_200 = SDIR + "CNN+NCP/all/seed_200/"

else:  # define folders in local PC
    pass

PIC_FOLDER = "rgb/data"
STEERING_PATH = "vehicle_data/steering.txt"
THROTTLE_PATH = "vehicle_data/throttle.txt"
BRAKE_PATH = "vehicle_data/brake.txt"

dev_cnn_s, dev_cnn_t, dev_cnn_b = [], [], []
dev_lstm_s, dev_lstm_t, dev_lstm_b = [], [], []
dev_gru_s, dev_gru_t, dev_gru_b = [], [], []
dev_ctgru_s, dev_ctgru_t, dev_ctgru_b = [], [], []
dev_ncp_s, dev_ncp_t, dev_ncp_b = [], [], []

S_DIM = (3, 66, 200)
A_DIM = 3
SEQ_LENGTH = 1  # for test
transform = None if isinstance(S_DIM, int) else \
    transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Lambda(lambda img:__crop(img, (10, 80), (500, 176))),
         transforms.Resize((66, 200)),
         transforms.ToTensor()]
    )

# CNN models, 1: seed 100, 2: seed 150, 3: seed 200.
cnn_1 = Convolution_Model(S_DIM, A_DIM)
cnn_2 = Convolution_Model(S_DIM, A_DIM)
cnn_3 = Convolution_Model(S_DIM, A_DIM)

# CNN+LSTM
conv_head_lstm1 = ConvolutionHead_Nvidia(S_DIM,
                                         SEQ_LENGTH,
                                         num_filters=32,
                                         features_per_filter=4)
conv_head_lstm2 = ConvolutionHead_Nvidia(S_DIM,
                                         SEQ_LENGTH,
                                         num_filters=32,
                                         features_per_filter=4)
conv_head_lstm3 = ConvolutionHead_Nvidia(S_DIM,
                                         SEQ_LENGTH,
                                         num_filters=32,
                                         features_per_filter=4)
lstm1 = LSTM_Model(conv_head=conv_head_lstm1)
lstm2 = LSTM_Model(conv_head=conv_head_lstm2)
lstm3 = LSTM_Model(conv_head=conv_head_lstm3)

# CNN+GRU

conv_head_gru1 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
conv_head_gru2 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
conv_head_gru3 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
gru1 = GRU_Model(conv_head=conv_head_gru1)
gru2 = GRU_Model(conv_head=conv_head_gru2)
gru3 = GRU_Model(conv_head=conv_head_gru3)

# CNN+CTGRU
conv_head_ctgru1 = ConvolutionHead_Nvidia(S_DIM,
                                          SEQ_LENGTH,
                                          num_filters=32,
                                          features_per_filter=4)
conv_head_ctgru2 = ConvolutionHead_Nvidia(S_DIM,
                                          SEQ_LENGTH,
                                          num_filters=32,
                                          features_per_filter=4)
conv_head_ctgru3 = ConvolutionHead_Nvidia(S_DIM,
                                          SEQ_LENGTH,
                                          num_filters=32,
                                          features_per_filter=4)
ctgru1 = CTGRU_Model(64,
                     conv_head=conv_head_ctgru1,
                     time_step=1,
                     use_cuda=False)  # 64 hidden units
ctgru2 = CTGRU_Model(64,
                     conv_head=conv_head_ctgru2,
                     time_step=1,
                     use_cuda=False)
ctgru3 = CTGRU_Model(64,
                     conv_head=conv_head_ctgru3,
                     time_step=1,
                     use_cuda=False)

# CNN+NCP
conv_head_ncp1 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
conv_head_ncp2 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
conv_head_ncp3 = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
input_shape = \
    (1, conv_head_ncp1.num_filters * conv_head_ncp1.features_per_filter)
wiring = NCPWiring(inter_neurons=64, command_neurons=32, motor_neurons=3,
                   sensory_fanout=48, inter_fanout=24,
                   recurrent_command=24, motor_fanin=16)
wiring.build(input_shape)
ltc_cell_1 = LTCCell(wiring=wiring, time_interval=0.04)
ltc_cell_2 = LTCCell(wiring=wiring, time_interval=0.04)
ltc_cell_3 = LTCCell(wiring=wiring, time_interval=0.04)

ncp1 = NCP_Model(ltc_cell=ltc_cell_1, conv_head=conv_head_ncp1)
ncp2 = NCP_Model(ltc_cell=ltc_cell_2, conv_head=conv_head_ncp2)
ncp3 = NCP_Model(ltc_cell=ltc_cell_3, conv_head=conv_head_ncp3)

# load params
cnn_1.load(CNN_S_100)
cnn_1.eval()
cnn_2.load(CNN_S_150)
cnn_2.eval()
cnn_3.load(CNN_S_200)
cnn_3.eval()

lstm1.load(LSTM_S_100)
lstm1.eval()
lstm2.load(LSTM_S_150)  # problem
lstm2.eval()
lstm3.load(LSTM_S_200)
lstm3.eval()

gru1.load(GRU_S_100)
lstm1.eval()
gru2.load(GRU_S_150)
lstm2.eval()
gru3.load(GRU_S_200)
lstm3.eval()

ctgru1.load(CTGRU_S_100)
lstm1.eval()
ctgru2.load(CTGRU_S_150)
lstm2.eval()
ctgru3.load(CTGRU_S_200)
lstm3.eval()

ncp1.load(NCP_S_100)
ncp1.eval()
ncp2.load(NCP_S_150)
ncp2.eval()
ncp3.load(NCP_S_200)
ncp3.eval()


# initialize variable to hold the deviation.

sequence = os.listdir(TESTDATA)  # how many sequences need to be visualized
sequence = list(filter(lambda a: '.DS_Store' not in a, sequence))
sequence = sorted(sequence)
# print(sequence)

# plot for only one sequence, iterate over the whole data,
for j in sequence:
    # reset in the beginning of every sequence.
    s_cnn, s_lstm, s_gru, s_ctgru, s_ncp, s_truth \
        = [], [], [], [], [], []
    # steering prediction of cnn,cnn+lstm,cnn+gru,cnn+ctgru,cnn+ncp

    t_cnn, t_lstm, t_gru, t_ctgru, t_ncp, t_truth \
        = [], [], [], [], [], []
    # throttle prediction of cnn,cnn+lstm,cnn+gru,cnn+ctgru,cnn+ncp

    b_cnn, b_lstm, b_gru, b_ctgru, b_ncp, b_truth \
        = [], [], [], [], [], []
    # throttle prediction of cnn,cnn+lstm,cnn+gru,cnn+ctgru,cnn+ncp

    PATH = TESTDATA + "/" + j
    img_path = os.path.join(PATH, PIC_FOLDER)
    list_imgs = sorted(os.listdir(img_path))
    hidden_state_lstm1, hidden_state_lstm2, hidden_state_lstm3 = \
        None, None, None
    hidden_state_gru1, hidden_state_gru2, hidden_state_gru3 = \
        None, None, None
    hidden_state_ctgru1, hidden_state_ctgru2, hidden_state_ctgru3 = \
        None, None, None
    hidden_state_ncp1, hidden_state_ncp2, hidden_state_ncp3 = \
        None, None, None

    for pic_name in list_imgs:
        if "timestamps" not in pic_name and \
                ".DS_Store" not in pic_name and "._" not in pic_name:
            pic_path = os.path.join(img_path, pic_name)
            # the rgb_image path
            states = np.asarray(Image.open(pic_path), dtype=np.uint8)
            states_cnn = transform(states)  # this is for CNN
            # input of RNN, should be B,T,C,H,W, needs to broadcast for use.
            states_rnn = torch.unsqueeze(states_cnn, 0)
            states_rnn = torch.unsqueeze(states_rnn, 0)
            # print(states_NCP.shape)
            # calculate average prediction of CNN
            actions_cnn_1 = cnn_1(states_cnn)
            actions_cnn_2 = cnn_2(states_cnn)
            actions_cnn_3 = cnn_3(states_cnn)
            actions_average = (actions_cnn_1+actions_cnn_2+actions_cnn_3)/3
            s_cnn.append(float(actions_average[0][0]))
            t_cnn.append(float(actions_average[0][1]))
            b_cnn.append(float(actions_average[0][2]))

            # calculate average prediction of CNN+LSTM
            actions_lstm_1, hidden_state_lstm1 = \
                lstm1.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_lstm1
                )
            actions_lstm_2, hidden_state_lstm2 = \
                lstm2.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_lstm2
                )
            actions_lstm_3, hidden_state_lstm3 = \
                lstm3.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_lstm3
                )
            actions_average =\
                (actions_lstm_1 + actions_lstm_2 + actions_lstm_3) / 3
            s_lstm.append(float(actions_average[0][0][0]))
            t_lstm.append(float(actions_average[0][0][1]))
            b_lstm.append(float(actions_average[0][0][2]))

            # calculate average prediction of CNN+GRU
            actions_gru_1, hidden_state_gru1 = \
                gru1.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_gru1
                )
            actions_gru_2, hidden_state_gru2 = \
                gru2.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_gru2
                )
            actions_gru_3, hidden_state_gru3 = \
                gru3.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_gru3
                )
            actions_average = \
                (actions_gru_1 + actions_gru_2 + actions_gru_3) / 3
            s_gru.append(float(actions_average[0][0][0]))
            t_gru.append(float(actions_average[0][0][1]))
            b_gru.append(float(actions_average[0][0][2]))

            # calculate average prediction of CNN+CTGRU
            actions_ctgru_1, hidden_state_ctgru1 = \
                ctgru1.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_ctgru1
                )
            actions_ctgru_2, hidden_state_ctgru2 = \
                ctgru2.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_ctgru2
                )
            actions_ctgru_3, hidden_state_ctgru3 = \
                lstm3.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_ctgru3
                )
            actions_average = \
                (actions_ctgru_1 + actions_ctgru_2 + actions_ctgru_3) / 3
            s_ctgru.append(float(actions_average[0][0][0]))
            t_ctgru.append(float(actions_average[0][0][1]))
            b_ctgru.append(float(actions_average[0][0][2]))

            # calculate average prediction of CNN+NCP
            actions_ncp_1, hidden_state_ncp1 = \
                ncp1.evaluate_on_single_sequence(
                    states_rnn, hidden_state=hidden_state_ncp1
                )
            actions_ncp_2, hidden_state_ncp2 = \
                ncp2.evaluate_on_single_sequence(
                    states_rnn,
                    hidden_state=hidden_state_ncp2
                )
            actions_ncp_3, hidden_state_ncp3 = \
                ncp3.evaluate_on_single_sequence(
                    states_rnn,
                    hidden_state=hidden_state_ncp3
                )
            actions_average = \
                (actions_ncp_1 + actions_ncp_2 + actions_ncp_3) / 3
            s_ncp.append(float(actions_average[0][0][0]))
            t_ncp.append(float(actions_average[0][0][1]))
            b_ncp.append(float(actions_average[0][0][2]))

    steering_path_full = os.path.join(PATH, STEERING_PATH)
    throttle_path_full = os.path.join(PATH, THROTTLE_PATH)
    brake_path_full = os.path.join(PATH, BRAKE_PATH)

    # read data of steering, throttle, brake
    steering_raw = pd.read_csv(steering_path_full,
                               header=None)[0].values.tolist()
    print(len(steering_raw))
    throttle_raw = pd.read_csv(throttle_path_full,
                               header=None)[0].values.tolist()
    print(len(throttle_raw))
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
    print(len(s_cnn))

    assert \
        len(s_truth) == len(s_cnn) == len(s_lstm) == \
        len(s_gru) == len(s_ctgru), \
        "check number of steering commands"
    assert \
        len(t_truth) == len(t_cnn) == len(t_lstm) == \
        len(t_gru) == len(t_ctgru), \
        "check number of throttle commands"
    assert \
        len(b_truth) == len(b_cnn) == len(b_lstm) == \
        len(b_gru) == len(b_ctgru), \
        "check number of brake commands"
    assert len(s_truth) == len(t_truth) == len(b_truth), "check csv file"

    # save the offline simulation data.
    make_dirs(RDIR + "offline_simulation/" + j + "/")
    # sequence idx needs change
    # draw the plots
    data1 = {"Ground_truth": s_truth,
             "model_1": s_cnn,
             "model_2": s_lstm,
             "model_3": s_gru,
             "model_4": s_ctgru,
             "model_5": s_ncp
             }
    data2 = {"Ground_truth": t_truth,
             "model_1": t_cnn,
             "model_2": t_lstm,
             "model_3": t_gru,
             "model_4": t_ctgru,
             "model_5": t_ncp
             }
    data3 = {"Ground_truth": b_truth,
             "model_1": b_cnn,
             "model_2": b_lstm,
             "model_3": b_gru,
             "model_4": b_ctgru,
             "model_5": b_ncp
             }
    # steering
    plt.clf()
    plt.title("Sequential plot of steering",
              fontsize=11,
              fontweight='bold')
    plt.xlabel("Time/s", fontsize=11)
    plt.ylabel("Steering", fontsize=11)
    plt.grid(ls='--')
    for key, val in data1.items():
        plt.plot(0.04 * np.arange(len(val)), val, label=key)
    plt.legend()
    plt.savefig(
        RDIR + "offline_simulation/" + j + "/steering.pdf")

    # throttle
    plt.clf()
    plt.title("Sequential plot of throttle",
              fontsize=11,
              fontweight='bold')
    plt.xlabel("Time/s", fontsize=11)
    plt.ylabel("Throttle", fontsize=11)
    plt.grid(ls='--')
    for key, val in data2.items():
        plt.plot(0.04 * np.arange(len(val)), val, label=key)
    plt.legend()
    plt.savefig(
        RDIR + "offline_simulation/" + j + "/throttle.pdf")

    # brake
    plt.clf()
    plt.title("Sequential plot of brake",
              fontsize=11,
              fontweight='bold')
    plt.xlabel("Time/s", fontsize=11)
    plt.ylabel("Brake", fontsize=11)
    plt.grid(ls='--')
    for key, val in data3.items():
        plt.plot(0.04 * np.arange(len(val)), val, label=key)
    plt.legend()
    plt.savefig(
        RDIR + "offline_simulation/" + j + "/brake.pdf")

    # calculate the absolute deviation of this sequence.

    # calculate deviation of cnn
    dev_cnn_s.extend(
        np.absolute((np.array(s_cnn)-np.array(s_truth))).tolist()
    )
    dev_cnn_t.extend(
        np.absolute((np.array(t_cnn)-np.array(t_truth))).tolist()
    )
    dev_cnn_b.extend(
        np.absolute((np.array(b_cnn) - np.array(b_truth))).tolist()
    )
    # calculate deviation of cnn+lstm
    dev_lstm_s.extend(
        np.absolute((np.array(s_lstm)-np.array(s_truth))).tolist()
    )
    dev_lstm_t.extend(
        np.absolute((np.array(t_lstm)-np.array(t_truth))).tolist()
    )
    dev_lstm_b.extend(
        np.absolute((np.array(b_lstm) - np.array(b_truth))).tolist()
    )
    # calculate deviation of cnn+gru
    dev_gru_s.extend(
        np.absolute((np.array(s_gru) - np.array(s_truth))).tolist()
    )
    dev_gru_t.extend(
        np.absolute((np.array(t_gru) - np.array(t_truth))).tolist()
    )
    dev_gru_b.extend(
        np.absolute((np.array(b_gru) - np.array(b_truth))).tolist()
    )
    # calculate deviation of cnn+ctgru
    dev_ctgru_s.extend(
        np.absolute((np.array(s_ctgru) - np.array(s_truth))).tolist()
    )
    dev_ctgru_t.extend(
        np.absolute((np.array(t_ctgru) - np.array(t_truth))).tolist()
    )
    dev_ctgru_b.extend(
        np.absolute((np.array(b_ctgru) - np.array(b_truth))).tolist()
    )
    # calculate deviation of cnn+ncp
    dev_ncp_s.extend(
        np.absolute((np.array(s_ncp) - np.array(s_truth))).tolist()
    )
    dev_ncp_t.extend(
        np.absolute((np.array(t_ncp) - np.array(t_truth))).tolist()
    )
    dev_ncp_b.extend(
        np.absolute((np.array(b_ncp) - np.array(b_truth))).tolist()
    )

# get all the absolute deviation, save in csv file.
DEVIATION = {'deviation of steering for CNN': dev_cnn_s,
             'deviation of steering for CNN+LSTM': dev_lstm_s,
             'deviation of steering for CNN+GRU': dev_gru_s,
             'deviation of steering for CNN+CTGRU': dev_ctgru_s,
             'deviation of steering for CNN+NCP': dev_ncp_s,
             'deviation of throttle for CNN': dev_cnn_t,
             'deviation of throttle for CNN+LSTM': dev_lstm_t,
             'deviation of throttle for CNN+GRU': dev_gru_t,
             'deviation of throttle for CNN+CTGRU': dev_ctgru_t,
             'deviation of throttle for CNN+NCP': dev_ncp_t,
             'deviation of brake for CNN': dev_cnn_b,
             'deviation of brake for CNN+LSTM': dev_lstm_b,
             'deviation of brake for CNN+GRU': dev_gru_b,
             'deviation of brake for CNN+CTGRU': dev_ctgru_b,
             'deviation of brake for CNN+NCP': dev_ncp_b
             }
pd.DataFrame(DEVIATION).to_csv(RDIR + "deviation.csv")
