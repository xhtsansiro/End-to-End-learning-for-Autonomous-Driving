"""This script measures the latency of all models.

Latency is the feed forward time of the models,
models: CNN, CNN+LSTM, CNN+GRU, CNN+CTGRU, CNN+NCP
"""

import time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from wiring import NCPWiring
from nets.ltc_cell import LTCCell
from nets.cnn_head import ConvolutionHead_Nvidia
from nets.models_all import Convolution_Model, LSTM_Model,\
    GRU_Model, CTGRU_Model, NCP_Model
from utils import __crop

CNN = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
      "08_Model_comparison/01_Trained_model/CNN/all/seed_200/"
LSTM = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
       "08_Model_comparison/01_Trained_model/CNN+LSTM/all/seed_200/"
GRU = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
      "08_Model_comparison/01_Trained_model/CNN+GRU/all/seed_200/"
CTGRU = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
      "08_Model_comparison/01_Trained_model/CNN+CTGRU/all/seed_200/"
NCP = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
      "08_Model_comparison/01_Trained_model/CNN+NCP/all/seed_200/"

IMAGE_PATH = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
            "06_CNN_head_comparison/04_Comparison/05_016_0009_image.png"


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

# CNN
cnn = Convolution_Model(S_DIM, A_DIM)
cnn.load(CNN)
cnn.eval()
# CNN+LSTM
conv_head_lstm = ConvolutionHead_Nvidia(S_DIM,
                                        SEQ_LENGTH,
                                        num_filters=32,
                                        features_per_filter=4)
lstm = LSTM_Model(conv_head=conv_head_lstm)
lstm.load(LSTM)
lstm.eval()
# CNN+GRU
conv_head_gru = ConvolutionHead_Nvidia(S_DIM,
                                       SEQ_LENGTH,
                                       num_filters=32,
                                       features_per_filter=4)
gru = GRU_Model(conv_head=conv_head_gru)
gru.load(GRU)
gru.eval()
# CNN+CTGRU
conv_head_ctgru = ConvolutionHead_Nvidia(S_DIM,
                                         SEQ_LENGTH,
                                         num_filters=32,
                                         features_per_filter=4)
ctgru = CTGRU_Model(64,
                    conv_head=conv_head_ctgru,
                    time_step=1,
                    use_cuda=False)  # 64 hidden units
ctgru.load(CTGRU)
ctgru.eval()
# CNN+NCP
conv_head_ncp = ConvolutionHead_Nvidia(S_DIM,
                                       SEQ_LENGTH,
                                       num_filters=32,
                                       features_per_filter=4)
input_shape = \
    (1, conv_head_ncp.num_filters * conv_head_ncp.features_per_filter)
wiring = NCPWiring(inter_neurons=64, command_neurons=32, motor_neurons=3,
                   sensory_fanout=48, inter_fanout=24,
                   recurrent_command=24, motor_fanin=16)
wiring.build(input_shape)
ltc_cell = LTCCell(wiring=wiring, time_interval=0.04)

ncp = NCP_Model(ltc_cell=ltc_cell, conv_head=conv_head_ncp)
ncp.load(NCP)
ncp.eval()

# image
states = np.asarray(Image.open(IMAGE_PATH), dtype=np.uint8)
states_cnn = transform(states)  # this is for CNN
# input of CNN_head, should be B,T,C,H,W, needs to broadcast for use.
states_rnn = torch.unsqueeze(states_cnn, 0)
states_rnn = torch.unsqueeze(states_rnn, 0)

hidden_state_lstm, hidden_state_gru, hidden_state_ctgru, hidden_state_ncp \
    = None, None, None, None

t_cnn, t_lstm, t_gru, t_ctgru, t_ncp = [], [], [], [], []  # save time

# start calculating time for feedforward process
for i in range(200):
    # time needed for CNN
    t_1 = time.time()
    actions = cnn(states_cnn)
    t_cnn.append(time.time()-t_1)

    # time needed for CNN+LSTM
    t_2 = time.time()
    actions_lstm, hidden_state_lstm = lstm.evaluate_on_single_sequence(
        states_rnn, hidden_state=hidden_state_lstm)
    t_lstm.append(time.time() - t_2)

    # time needed for CNN+GRU
    t_3 = time.time()
    actions_gru, hidden_state_gru = gru.evaluate_on_single_sequence(
        states_rnn, hidden_state=hidden_state_gru)
    t_gru.append(time.time() - t_3)

    # time needed for CNN+CTGRU
    t_4 = time.time()
    actions_ctgru, hidden_state_ctgru = ctgru.evaluate_on_single_sequence(
        states_rnn, hidden_state=hidden_state_ctgru)
    t_ctgru.append(time.time() - t_4)

    # time needed for CNN+NCP
    t_5 = time.time()
    actions_ncp, hidden_state_ncp = ncp.evaluate_on_single_sequence(
        states_rnn, hidden_state=hidden_state_ncp)
    t_ncp.append(time.time() - t_5)

print(t_cnn)
print('-----')
print(t_lstm)
print('-----')
print(t_gru)
print('-----')
print(t_ctgru)
print('-----')
print(t_ncp)
print('-----')

TIME_CNN = f"Feedforward time for CNN {sum(t_cnn)/200}; "
TIME_LSTM = f"Feedforward time for CNN+LSTM {sum(t_lstm)/200}; "
TIME_GRU = f"Feedforward time for CNN+GRU {sum(t_gru)/200}; "
TIME_CTGRU = f"Feedforward time for CNN+CTGRU {sum(t_ctgru)/200}; "
TIME_NCP = f"Feedforward time for CNN+NCP {sum(t_ncp)/200}; "


F = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
    "08_Model_comparison/latency.txt"
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(TIME_CNN)
    fil.write(TIME_LSTM)
    fil.write(TIME_GRU)
    fil.write(TIME_CTGRU)
    fil.write(TIME_NCP)
