"""Test feed-forward time.

This file is used to test feed-forward
time of trained model for CNN head.
comment line 641,642 of models_all,
uncomment line 644-646 of models_all.
"""
import time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from wiring import NCPWiring
from nets.ltc_cell import LTCCell
from nets.cnn_head import ConvolutionHead_Nvidia, ConvolutionHead_AlexNet, \
    ConvolutionHead_ResNet
from nets.models_all import NCP_Model
from utils import __crop

NCP_N_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
            "06_CNN_head_comparison/05_Trained_model/" \
            "01_NvidiaCNN+NCP/seed_100/"
NCP_A_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
            "06_CNN_head_comparison/05_Trained_model/" \
            "02_AlexNet+NCP/seed_100/"
NCP_R_100 = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
            "06_CNN_head_comparison/05_Trained_model/" \
            "03_ResNet+NCP/seed_100/"
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

input_shape = (1, 32 * 4)  # the same for every cnn head.
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

policy_n.load(NCP_N_100)
policy_n.eval()  # in evaluation mode

policy_a.load(NCP_A_100)
policy_a.eval()  # in evaluation mode

policy_r.load(NCP_R_100)
policy_r.eval()  # in evaluation mode

# image
states = np.asarray(Image.open(IMAGE_PATH), dtype=np.uint8)
states_CNN = transform(states)  # this is for CNN
# input of CNN_head, should be B,T,C,H,W, needs to broadcast for use.
states_NCP = torch.unsqueeze(states_CNN, 0)
states_NCP = torch.unsqueeze(states_NCP, 0)

hidden_state_n, hidden_state_a, hidden_state_r = None, None, None
t_n, t_a, t_r = [], [], []  # save time,
# t_n: time_nvidia, t_a: time_alexNet, t_r: time_resnet

# start calculating time for feedforward process
for i in range(200):

    t_1 = time.time()
    act_pairs_r, hidden_state_r = policy_r.evaluate_on_single_sequence(
        states_NCP,
        hidden_state=hidden_state_r
    )
    t_r.append(time.time() - t_1)

    t_2 = time.time()
    act_pairs_a, hidden_state_a = policy_a.evaluate_on_single_sequence(
        states_NCP,
        hidden_state=hidden_state_a
    )
    t_a.append(time.time() - t_2)

    t_3 = time.time()
    act_pairs_n, hidden_state_n = policy_n.evaluate_on_single_sequence(
        states_NCP,
        hidden_state=hidden_state_n
    )
    t_n.append(time.time() - t_3)

print(t_r)
print('---')
print(t_a)
print('---')
print(t_n)

TIME_1 = f"Average Feedforward time for ResNet+NCP {sum(t_r)/200}; "
TIME_2 = f"Average Feedforward time for AlexNet+NCP {sum(t_a)/200}; "
TIME_3 = f"Average Feedforward time for NvidiaCNN+NCP {sum(t_n)/200}; "

F = "/home/ubuntu/repos/results_haotian/01_CARLA_Data/" \
            "06_CNN_head_comparison/latency.txt"
with open(F, 'w', encoding='utf-8', errors='surrogateescape') as fil:
    fil.write(TIME_1)
    fil.write(TIME_2)
    fil.write(TIME_3)
