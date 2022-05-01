"""This script defines all the models except NCP used in this work."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from utils import _image_standardization
from nets.cnn_head import ConvolutionHead_Nvidia


class Convolution_Model(nn.Module):
    """This class defines CNN baseline."""

    def __init__(self, img_dim, a_dim, use_cuda=True):
        """Initialize the object."""
        super(Convolution_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')
        self.conv = nn.Sequential(  # input is (3,66,200)
            nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=True),  # (3198)
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=True),  # (14,47)
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=True),  # (5,22)
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=True),  # (3,20)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),  # (1,18)
            nn.ReLU(inplace=True)
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(1152, 100)
        # need to adjust based on the input shape of the img.
        self.linear2 = nn.Linear(100, 50)   # extra added
        self.linear3 = nn.Linear(50, 10)    # extra added
        self.linear4 = nn.Linear(10, a_dim)  # (10,1) predict only steering.
        # self.a_dim = a_dim
        self.channel = img_dim[0]
        self.height = img_dim[1]
        self.width = img_dim[2]
        # self.num_params = 0

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=1e-4)  # use adam optimizer
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        # self.optimizer, milestones=[10], gamma=0.4, verbose=True)

    def forward(self, x):
        """Define forward process of CNN baseline."""
        # pic standardization
        x = _image_standardization(x)
        x = x.view(-1, self.channel, self.height, self.width)
        # go to conv layer
        x = self.conv(x)  # output is (N, C, H, W)
        x = x.view(-1, 64*18)
        x = self.dropout1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout2(x)
        x = F.relu(self.linear2(x))
        x = self.dropout3(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def criterion(self, a_imitator, a_exp):
        """Define L2 loss."""
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """Define weighted loss."""
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """Save the trained model."""
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        """Load the trained model."""
        try:
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("load parameters are in" + ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

    def count_params(self):
        """Count how many learnable parameters are there in the network."""
        num_params = sum(param.numel() for param in self.parameters())
        return num_params

    def nn_structure(self):
        """Save network layer info. into a dic, and return the dict."""
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class GRU_Model(nn.Module):
    """This class defines GRU model, layer is equal to 1."""

    # the output of cnn_head decides the input_size
    def __init__(self, conv_head,
                 time_step=16, input_size=128, hidden_size=64,
                 output=3, use_cuda=True):
        """Initialize the object."""
        super(GRU_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")

        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1
        self.num_params = 0

        self.conv_head = conv_head  # cnn layer before RNN
        self.time_step = time_step
        # the number of expected features in input
        self.input_size = input_size
        # the number of features in hidden state h,
        # which should be the size of the output
        self.hidden_size = hidden_size
        self.output = output
        self.GRU = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        # default: num_layers = 1
        self.linear = nn.Linear(self.hidden_size, self.output)  # linear(64,3)
        # optimizer should be declared after the def of layer,
        # so that self.parameters() is not empty.
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        # self.optimizer = optim.Adam(
        # [{'params': self.conv_head.parameters(), 'lr': 1e-4},
        # from 2.5e-5 to 1e-4
        # {'params': self.GRU.parameters()},
        # {'params': self.linear.parameters()}], lr=5e-4)

    def forward(self, x):
        """Define forward process of GRU."""
        # because the batch size can be different for the last batch,
        # so not initialized in __init__
        batch_size = x.shape[0]
        x = self.conv_head(x)
        # after conv head, the tensor has the shape (N, T, F),
        # batch_size * time sequence * features
        x, _ = self.GRU(x)
        # (h_0, c_0 default to zero if not provided),
        # (h_T, c_T) represent cell state/hidden state of last size

        x = x.contiguous().view(-1, self.hidden_size)
        x = self.linear(x)  # mapping hidden_size 64 to output dimension 3,
        x = x.view(batch_size, self.time_step, self.output)
        return x

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """Evaluate whole sequence sequentially, for test or valid."""
        x = self.conv_head(x)
        # shape (BatchSize, Time_Sequence, ExtractedFeatures)

        # initial hidden state has the shape (D*num_layers, N, H_out)
        # N: Batch size, L: sequence length
        if hidden_state is None:
            hidden_state = torch.zeros((1, x.shape[0], self.hidden_size),
                                       device=x.device)
        # there will be a update, x: (N,L,H_out), hidden_state: (D,N,H_out)
        # in this case, N=1, L=16,
        result, hidden_state = self.GRU(x, hidden_state)
        result = result.view(-1, self.hidden_size)

        # map the hidden out to the action pair
        result = self.linear(result)

        result = torch.unsqueeze(result, dim=0)  # map to [1,16,3]
        return result, hidden_state

    def criterion(self, a_imitator, a_exp):
        """Calculate original loss."""
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """Calculate weighted loss."""
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, SDIR):
        """Save the model after training."""
        torch.save(self.state_dict(), SDIR + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), SDIR + "policy_optim.pth")

    def load(self, LDIR):
        """Load the trained model."""
        try:
            self.load_state_dict(
                torch.load(LDIR + "policy_model.pth",
                           map_location=torch.device('cpu')
                           )
            )
            self.optimizer.load_state_dict(
                torch.load(LDIR + "policy_optim.pth",
                           map_location=torch.device('cpu'))
            )
            print("load parameters are in" + LDIR)
            return True
        except:
            print("parameters are not loaded")
            return False

    def count_params(self):
        """Count how many learnable parameters are there in the network."""
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """Get structure of the model."""
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


# LSTM implemented, layer number: 1
class LSTM_Model(nn.Module):
    """This class defines the lSTM, layer is equal to 1."""

    # the output of cnn_head decides the input_size
    def __init__(self, conv_head,
                 time_step=16, input_size=128, hidden_size=64,
                 output=3, use_cuda=True):
        """Initialize the object."""
        super(LSTM_Model, self).__init__()

        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')
        self.exp_factor = 0.1
        self.num_params = 0

        self.conv_head = conv_head   # cnn layer before LSTM
        self.time_step = time_step

        # the number of expected features in input
        self.input_size = input_size

        # the number of features in hidden state h,
        # which should be the size of the output
        self.hidden_size = hidden_size
        self.output = output
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            batch_first=True)  # default: num_layers = 1
        self.linear = nn.Linear(self.hidden_size, self.output)  # linear(64,3)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):  # no necessary to define h_0 and c_0 explicitly.
        """Define forward process for LSTM."""
        batch_size = x.shape[0]
        x = self.conv_head(x)
        # after conv head, the tensor has the shape (N, T, F),
        # batch_size * time sequence * features

        x, _ = self.lstm(x)
        # (h_0, c_0 default to zero if not provided),
        # (h_T, c_T) represent cell state/hidden state of last size

        # x has the shape (batch_size, time_step, hidden_size),
        # use a fully connected layer to transfer to the output dimension 3
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.linear(x)  # mapping hidden_size 64 to output dimension 3,

        x = x.view(batch_size, self.time_step, self.output)
        # x = torch.squeeze(x)
        return x

    # evaluate step by step, time_sequence = 1,
    # for LSTM. the hidden state is (h, c), H_cell = H_h
    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """Evaluate whole sequence sequentially, for valid or test."""
        x = self.conv_head(x)
        # shape (BatchSize, Time_Sequence, ExtractedFeatures)

        if hidden_state is None:
            hidden_state = (torch.zeros((1, x.shape[0], self.hidden_size),
                                        device=x.device),
                            torch.zeros((1, x.shape[0], self.hidden_size),
                                        device=x.device)
                            )
        # h_0 and c_0 has the same shape, hidden_state = (h0,c0)

        result, hidden_state = self.lstm(x, hidden_state)
        # there will be a update, x: (N,L,H_out).

        # in this case, N=1, L=16 (1,16,64) to (16,64)
        result = result.view(-1, self.hidden_size)
        result = self.linear(result)

        result = torch.unsqueeze(result, dim=0)  # map to [1,16,3]
        return result, hidden_state

    def criterion(self, a_imitator, a_exp):
        """Calculate original loss."""
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """Calculate weighted loss."""
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """Save the model after training."""
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    # load the trained model
    def load(self, ldir):
        """Load the trained model."""
        try:
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("load parameters are in" + ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

    def count_params(self):
        """Count how many learnable parameters are there in the network."""
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """Get the structure of the model."""
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class CTGRU_Model(nn.Module):
    """This class defines CT-GRU."""

    def __init__(self, num_units, conv_head,
                 M=8, time_step=16, output=3, use_cuda=True):
        """Initialize the object."""
        super(CTGRU_Model, self).__init__()
        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        self.loss = nn.MSELoss(reduction='mean')

        self.linear = nn.Linear(num_units, 3)
        # hidden_neurons  --> output (steering, brake, throttle).

        self.conv_head = conv_head
        self._num_units = num_units
        # the hidden size, h1, h2, .... h_N (for each input)

        self.M = M   # how many traces

        ln_tau_table = np.empty(self.M)
        # store the time_constant of the traces

        self.time_step = time_step
        self.output = output
        tau = 1
        for i in range(self.M):
            ln_tau_table[i] = np.log(tau)
            tau = tau * (10 ** 0.5)
        # tao_i+1 = 10^0.5 * tao_i delivers high fidelity

        self.ln_tau_table = torch.tensor(
            ln_tau_table, dtype=torch.float32, device=self.device)
        # initialize some variables here

        self.fused_input = None
        # (x, h_k-1), shape (BS, input_size + hidden_size)

        self.ln_tau_r = None
        self.ln_tau_s = None
        self.sf_input_r = None
        self.r_ki = None
        self.s_ki = None
        self.fused_q_input = None
        self.q_k = None
        self.sf_input_s = None
        self.delta_t = torch.tensor(0.04, device=self.device)
        # pic interval, 0.04 for training, 0.2 for simulation test

        # how many features does one sample have, from conv_head
        self.feature_number = \
            self.conv_head.num_filters * self.conv_head.features_per_filter
        self.linear_r = nn.Linear(
            self.feature_number + self._num_units,
            self._num_units * M
        )  # calculate ln_tau_r
        self.linear_q = nn.Linear(
            self.feature_number + self._num_units,
            self._num_units
        )  # calculate q
        self.linear_s = nn.Linear(
            self.feature_number + self._num_units,
            self._num_units * M
        )  # calculate ln_tau_s
        self.linear = nn.Linear(
            num_units,
            self.output
        )  # hidden_neurons-->(steering, brake, throttle)
        self.softmax_r = nn.Softmax(dim=2)
        self.softmax_s = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        #  self._allocate()  # allocate the variables
        self.num_params = 0
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=1e-4,
            weight_decay=0
        )  # weight_decay default 0.

    @property
    def state_size(self):
        """Define the state size."""
        return self._num_units * self.M  # hidden_size * trace_number

    @property
    def output_size(self):
        """Define the hidden size."""
        return self._num_units  # hidden size

    def update(self, x, h_hat, delta_t):
        """Update the states in one time interval."""
        # x is the input at each time point (BS, Features)
        # which is h_hat, h_hat has the shape (BS, num_units, M),
        # h_k = sum (h_hat_ki) over all i

        h = h_hat.sum(dim=2)
        # sum over all traces for one element of hidden_units of one sample,
        # (BS, num_units)

        # determine retrieval scale and weighting
        # ln_tao_r = w * x +  u * h_k-1 + b, and r_ki
        self.fused_input = torch.cat([x, h], dim=1)
        # (BS, input feature + num_units)
        self.ln_tau_r = self.linear_r(self.fused_input)
        # (BS, num_units * M )
        self.ln_tau_r = self.ln_tau_r.view(-1, self._num_units, self.M)
        # (BS, num_units, M)
        self.sf_input_r = - torch.square(self.ln_tau_r - self.ln_tau_table)
        self.r_ki = self.softmax_r(self.sf_input_r)
        # determine relevant event signals q_k,
        # (BS, input features + num_units)
        self.fused_q_input = torch.cat(
            [x, torch.sum(self.r_ki * h_hat, dim=2)], dim=1)
        self.fused_q_input = self.linear_q(self.fused_q_input)
        self.q_k = self.tanh(self.fused_q_input)
        self.q_k = self.q_k.reshape(-1, self._num_units, 1)
        # in order to broadcast (32, 64, 1)

        # determine storage scale and weighting
        self.ln_tau_s = self.linear_s(self.fused_input)
        # (BS, num_units * M )
        self.ln_tau_s = self.ln_tau_s.view(-1, self._num_units, self.M)
        # (BS, num_units, M)
        self.sf_input_s = - torch.square(self.ln_tau_s - self.ln_tau_table)
        self.s_ki = self.softmax_s(self.sf_input_s)
        # calculate h_hat_next
        # print(self.s_ki.shape)
        # print(h_hat.shape)
        # print(self.q_k.shape)
        h_hat_next = \
            ((1 - self.s_ki) * h_hat + self.s_ki * self.q_k) * \
            torch.exp(-delta_t / (self.ln_tau_table + 1e-7))
        # + 1e-7, avoid dividing by zero

        # combine time_scales (different traces)
        hidden_state = torch.sum(h_hat_next, dim=2)  # (BS, num_units)

        return h_hat_next, hidden_state

    def forward(self, x):
        """Define forward process of CTGRU."""
        # batch_size = x.shape[0]
        # because the batch size can be different for the last batch,
        # so not initialized in __init__
        x = self.conv_head(x)
        # has shape (batch_size, time_Sequence, Features )

        h_hat = torch.zeros((x.shape[0], self._num_units, self.M),
                            device=x.device)
        # initial states all zeros, important is to initialize
        # this variable on the system device.
        outputs = []
        for t in range(self.time_step):
            inputs = x[:, t, :]
            # take the output from the CNN at each time_step (BS, Features)
            h_hat, hidden_state = self.update(inputs, h_hat, self.delta_t)
            # time interval between each img
            outputs.append(hidden_state)

        outputs = torch.stack(outputs, dim=1)
        # (BS, time_sequence, num_units)

        outputs = outputs.view(-1, self._num_units)
        outputs = self.linear(outputs)  # (BS, 3* time_Sequence)
        # the output should have the shape (batch_size, sequence, commands)
        outputs = outputs.view(-1, self.time_step, self.output)
        # print("one RNN forward is finished")
        return outputs

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """Evaluate the whole sequence sequentially."""
        x = self.conv_head(x)
        # shape (BatchSize, Time_Sequence, ExtractedFeatures),
        # should be (1,time_Sequence/?,num_features)
        results = []
        if hidden_state is None:
            # initial states all zeros,
            # important is to initialize this variable on the system device.
            hidden_state = torch.zeros(
                (x.shape[0], self._num_units, self.M), device=x.device
            )

        for t in range(self.time_step):
            inputs = x[:, t, :]
            # take the output from the CNN at each time_step (BS, Features)
            hidden_state, result = self.update(
                inputs, hidden_state, self.delta_t)
            # time interval between each img
            results.append(result)

        results = torch.stack(results, dim=1)
        # (BS, time_sequence, num_units)
        results = results.view(-1, self._num_units)
        # result has the shape (B,num_units)
        results = self.linear(results)
        # map the hidden out to the action pair (num_units, 3)
        results = results.view(-1, self.time_step, self.output)
        return results, hidden_state

    def criterion(self, a_imitator, a_exp):
        """Calculate original loss."""
        loss = self.loss(a_imitator, a_exp)
        return loss

    def weighted_criterion(self, a_imitator, a_exp):
        """Calculate weighted loss."""
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def release(self, sdir):
        """Save the model after training."""
        torch.save(self.state_dict(), sdir + "policy_model.pth")
        torch.save(self.optimizer.state_dict(), sdir + "policy_optim.pth")

    def load(self, ldir):
        """Load the trained model."""
        try:
            print("load parameters in CPU")
            self.load_state_dict(
                torch.load(
                    ldir + "policy_model.pth",
                    map_location=torch.device('cpu')
                )
            )
            self.optimizer.load_state_dict(
                torch.load(
                    ldir + "policy_optim.pth",
                    map_location=torch.device('cpu')
                )
            )
            print("load parameters are in" + ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

    def count_params(self):
        """Count how many learnable parameters are there in the network."""
        self.num_params = sum(param.numel() for param in self.parameters())
        return self.num_params

    def nn_structure(self):
        """Get the structure of the model."""
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


class NCP_Model(nn.Module):
    """This class combines ltc_neurons, wiring shape to form NCP."""

    def __init__(self, ltc_cell, conv_head, use_cuda=True):
        """Initialize the object."""
        super(NCP_Model, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

        self.device = torch.device("cuda"
                                   if use_cuda and torch.cuda.is_available()
                                   else "cpu")
        #  print(self.device) # print out which device will be used
        self.conv_head = conv_head
        self.ltc_cell = ltc_cell

        # for training and test
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=1e-4)
        # for CNN head comparison
        # self.optimizer = optim.Adam(
        #    [{'params': self.conv_head.parameters(), 'lr': 1e-4},
        #     {'params': self.ltc_cell.parameters()}], lr=5e-4)

        self.exp_factor = 0.1  # weight factor of the loss
        self.time_sequence = conv_head.time_sequence

    def forward(self, x):
        """Define forward process of the network."""
        # x has the shape (batch_size, time_sequence,channel,height, width)
        # first feed into convolution head
        y = self.conv_head(x)
        # the output has shape(batch_size,time_sequence,features)
        # then feed into the ltc_cell
        # initial states all zeros,
        # important is to initialize this variable on the system device.
        hidden_state = torch.zeros(
            (y.shape[0], self.ltc_cell.state_size), device=x.device)
        outputs = []
        for t in range(self.time_sequence):
            inputs = y[:, t, :]
            # take the output from the CNN at each time_step
            output, hidden_state = self.ltc_cell(inputs, hidden_state)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        # the output should have the shape (batch_size, sequence, commands)
        outputs = outputs.view(outputs.shape[0], self.time_sequence, -1)
        # print("one RNN forward is finished")
        return outputs

    def evaluate_on_single_sequence(self, x, hidden_state=None):
        """Evaluate whole sequence sequentially."""
        # x has the shape (16,channel, width, height)
        y = self.conv_head(x)   # shape of y (1,16,feature number)
        if hidden_state is None:
            hidden_state = torch.zeros(
                (y.shape[0], self.ltc_cell.state_size), device=x.device)

        outputs = []
        for t in range(self.time_sequence):
            inputs = y[:, t, :]
            # take the output from the CNN at each time_step
            output, hidden_state = self.ltc_cell(inputs, hidden_state)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        # (BS, Features) concat in dim=1, which is the dim of Sequnece.
        # --> (B,T,F)
        # outputs = outputs.view(outputs.shape[0], self.time_sequence, -1)
        # the output should have the shape
        # (batch_size, sequence, commands)
        # print("one RNN forward is finished")
        return outputs, hidden_state
        # hidden_state is used as input for the next 16_sequences

    # calculate the loss, for gradient-descent
    def weighted_criterion(self, a_imitator, a_exp):
        """Calculate weighted loss."""
        assert self.exp_factor >= 0
        weights = torch.exp(torch.abs(a_exp) * self.exp_factor)
        # exp (|y|* alpha)
        error = a_imitator - a_exp
        return torch.sum(weights * torch.square(error)) / torch.sum(weights)

    def criterion(self, a_imitator, a_exp):
        """Calculate the origin loss."""
        loss = self.loss(a_imitator, a_exp)
        return loss

    def release(self, sdir):
        """Save trained model."""
        torch.save(self.state_dict(),
                   sdir + "policy" + "_model.pth"
                   )
        torch.save(self.optimizer.state_dict(),
                   sdir + "policy" + "_optim.pth"
                   )
    # load the model

    def load(self, ldir):
        """Load trained model."""
        try:
            print("load parameters in CPU")
            self.load_state_dict(torch.load(
                ldir + "policy_model.pth",
                map_location=torch.device('cpu'))
            )
            self.optimizer.load_state_dict(torch.load(
                ldir + "policy_optim.pth",
                map_location=torch.device('cpu'))
            )

            print("load parameters are in" + ldir)
            return True
        except:
            print("parameters are not loaded")
            return False

    def count_params(self):
        """Count how many params of NCP needs to be trained."""
        num_of_synapses = self.ltc_cell.sensory_synpase_count
        # int(np.sum(np.abs(self._sensory_adjacency_matrix)
        num_of_sensory_synapses = self.ltc_cell.synapse_count

        # num_of_sensory_synapses =
        # int(np.sum(np.abs(self._sensory_adjacency_matrix)

        total_parameters = 0
        total_parameters += 3 * self.ltc_cell.state_size
        # count cm,gleak,vleak
        # Each synapse has Erev, W, mu and sigma as parameters
        # (i.e 4 in total)
        total_parameters += 4 * (num_of_sensory_synapses + num_of_synapses)

        return total_parameters

    # return structure of the neural network
    def nn_structure(self):
        """Get the layer structure of this model."""
        dict_layer = {i: self._modules[i] for i in self._modules.keys()}
        return dict_layer


if __name__ == "__main__":
    s = (3, 128, 256)
    a = 3
    policy1 = Convolution_Model(s, a)  # initialize the CNN model
    cnn = ConvolutionHead_Nvidia(s, 16, num_filters=32, features_per_filter=4)
    policy2 = LSTM_Model(conv_head=cnn)
