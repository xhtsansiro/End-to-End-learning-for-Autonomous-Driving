"""This script defines the ltc_cell and ncp_cell."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """This class defines LTC neuron."""

    def __init__(self, wiring, time_interval,
                 in_features=None, input_mapping="affine",
                 output_mapping="affine", ode_unfolds=6, epsilon=1e-8):
        """Initialize the object."""
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(None, in_features)
        if not wiring.is_built():
            raise ValueError("Wiring error, "
                             "unknown number of input features, "
                             "Please pass the parameter 'in_features' or "
                             "call the 'wiring.build()'.")

        self._init_ranges = {
            "gleak": (0.001, 1.0),  # leakage conductance
            "vleak": (-0.2, 0.2),  # resting potential of each neuron
            "cm": (0.4, 0.6),  # membrane capacitance
            "w": (0.001, 1.0),  # synapse weight
            "sigma": (3, 8),   # gamma_ij
            "mu": (0.3, 0.8),   # mu_ij
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        self._params = {}  # store the params
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        # should be the sampling time, means the time interval
        # between two images in a sequence.
        self.elapsed_time = time_interval  # 0.04s for the zurich Data
        # initialize the parameters
        self._allocate_parameters()

    @property
    def state_size(self):
        """Return the number of ltc neurons."""
        return self._wiring.units

    @property
    def sensory_size(self):
        """Return the number of sensory neurons."""
        return self._wiring.input_dim

    @property
    def motor_size(self):
        """Return the number of motor neurons."""
        return self._wiring.output_dim

    @property
    def output_size(self):
        """Return the number of outputs."""
        return self.motor_size

    @property
    def sensory_synpase_count(self):
        """Return the total connections between sensor neuron and others."""
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix))

    @property
    def synapse_count(self):
        """Return the total connections among ltc neurons."""
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    # add the parameters of the neurons in the model.params,
    # in order to be trainable
    def add_weight(self, name, init_value):
        """Add the parameters of the neurons in the model.params."""
        param = torch.nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        """Initialize the initial values randomly."""
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval-minval) + minval

    def _allocate_parameters(self):
        """Allocate parameters."""
        print("start allocate parameters of neurons!")
        # self._params = {}
        self._params["gleak"] = self.add_weight(
            name="gleak",
            init_value=self._get_init_value((self.state_size,), "gleak")
        )

        self._params["vleak"] = self.add_weight(
            name="vleak",
            init_value=self._get_init_value((self.state_size,), "vleak")
        )

        self._params["cm"] = self.add_weight(
            name="cm",
            init_value=self._get_init_value((self.state_size,), "cm")
        )

        self._params["sigma"] = self.add_weight(
            name="sigma",
            init_value=self._get_init_value(
                (self.state_size, self.state_size),
                "sigma"
            )
        )

        self._params["mu"] = self.add_weight(
            name="mu",
            init_value=self._get_init_value(
                (self.state_size, self.state_size),
                "mu"
            )
        )

        self._params["w"] = self.add_weight(
            name="w",
            init_value=self._get_init_value(
                (self.state_size, self.state_size),
                "w"
            )
        )

        self._params["erev"] = self.add_weight(
            name="erev",
            init_value=torch.Tensor(self._wiring.erev_initializer())
        )

        self._params["sensory_sigma"] = self.add_weight(
            name="sensory_sigma",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size),
                "sensory_sigma"
            )
        )

        self._params["sensory_mu"] = self.add_weight(
            name="sensory_mu",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size),
                "sensory_mu"
            )
        )

        self._params["sensory_w"] = self.add_weight(
            name="sensory_w",
            init_value=self._get_init_value(
                (self.sensory_size, self.state_size),
                "sensory_w"
            )
        )

        self._params["sensory_erev"] = self.add_weight(
            name="sensory_erev",
            init_value=torch.Tensor(self._wiring.sensory_erev_initializer()))

        # following params are not trainable params
        self._params["sparsity_mask"] = torch.Tensor(np.abs(
            self._wiring.adjacency_matrix)
        )
        self._params["sensory_sparsity_mask"] = torch.Tensor(np.abs(
            self._wiring.sensory_adjacency_matrix)
        )

        # y = wx + b for each feature, then y will be the input for the NCP
        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight(
                name="input_w",
                init_value=torch.ones((self.sensory_size,))
            )
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight(
                name="input_b",
                init_value=torch.zeros((self.sensory_size,))
            )

        # y = wx +b for each motor signal
        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight(
                name="output_w",
                init_value=torch.ones((self.motor_size,))
            )
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight(
                name="output_b",
                init_value=torch.zeros((self.motor_size,))
            )

    def _sigmoid(self, v_pre, mu, sigma):
        """Calculate delta(x_j)."""
        v_pre = torch.unsqueeze(v_pre, -1)
        x = sigma * (v_pre-mu)
        return torch.sigmoid(x)

    # ode_solver to update the state, using semi-implicit euler approach
    def _ode_solver(self, inputs, state, elapsed_time):
        """Update the state with ode_solver."""
        v_pre = state

        # for inter_neurons, the pre-synapse neuron are sensory neurons,
        # the state of sensory neuron is the inputs
        # inputs which are after mapping(affine).
        sensory_w_activation = \
            self._params["sensory_w"] * self._sigmoid(
                inputs,
                self._params["sensory_mu"],
                self._params["sensory_sigma"]
            )

        sensory_w_activation *= self._params["sensory_sparsity_mask"].\
            to(sensory_w_activation.device)
        # w_ij * sigma_i(x_j)
        # w is a random created matrix, non-zeros entries where
        # actually exist no connection should be zero.
        # w_ij * sigma_i(x_j) * E_ij
        sensory_rev_activation = \
            sensory_w_activation * self._params["sensory_erev"]

        # print("check0 {}".format(sensory_rev_activation.size()))
        # 分子 (batch_size, sensory_neurons, hidden_neurons)
        sensory_numerator = torch.sum(sensory_rev_activation, dim=1)
        # 分母
        sensory_denominator = torch.sum(sensory_w_activation, dim=1)

        # cm_t, it is cm/delta_t
        cm_t = self._params["cm"] / (elapsed_time/self._ode_unfolds)

        # one input is corresponding to the six steps of hidden_neuron state update
        for t in range(self._ode_unfolds):
            w_activation = \
                self._params["w"] * self._sigmoid(
                    v_pre,
                    self._params["mu"],
                    self._params["sigma"]
                )
            w_activation *= self._params["sparsity_mask"].\
                to(w_activation.device)

            rev_activation = w_activation * self._params["erev"]
            # (batch_size, hidden_neurons, hidden_neurons)

            w_numerator = \
                torch.sum(rev_activation, dim=1) + sensory_numerator
            w_denominator = \
                torch.sum(w_activation, dim=1) + sensory_denominator

            numerator = cm_t * v_pre + self._params["gleak"] * self._params["vleak"] * w_numerator
            denominator = cm_t + self._params["gleak"] + w_denominator

            # avoid dividing by 0
            v_pre = numerator/(denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs):
        """Map image features to input to inter-neurons."""
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]

        return inputs

    def _map_outputs(self, state):
        """Map state of motor neurons to final outputs."""
        output = state[:, 0:self.motor_size]
        # take the state of motor_neurons (batch_size, state_size)
        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._input_mapping == "affine":
            output = output + self._params["output_b"]

        return output

    # some parameters need to be bounded, cannot be smaller than 0
    def clip(self, w):
        """Bound the neuron parameters."""
        return F.relu(w)

    # synapse weight cannot be negative,
    # (exictation, inhibitation depends on the polarity)
    def apply_weight_constraints(self):
        """Apply the bounding."""
        self._params["w"].data = self.clip(self._params["w"].data)
        self._params["sensory_w"].data = \
            self.clip(self._params["sensory_w"].data)
        self._params["cm"].data = \
            self.clip(self._params["cm"].data)
        self._params["gleak"].data = \
            self.clip(self._params["gleak"].data)

    def forward(self, inputs, states):
        """Define the forward process."""
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, self.elapsed_time)
        outputs = self._map_outputs(next_state)
        return outputs, next_state
