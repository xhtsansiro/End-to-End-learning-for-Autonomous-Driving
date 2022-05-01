"""This script defines how layers of NCP are connected."""
import numpy as np
# from nets.CNN_head import ConvolutionHead_Nvidia


class Wiring:
    """This is class of how neurons are generally wired."""

    def __init__(self, units):
        """Initialize an object of this class."""
        # units is the number of total hidden neurons
        self.units = units
        # save the polarity of synpases
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.input_dim = None  # the input features
        self.output_dim = None  # the output commands
        self.sensory_adjacency_matrix = None

    def is_built(self):
        """Check if the input_dim is given."""
        # if self.input_dim is None, the return value is false.
        return self.input_dim is not None

    def build(self, input_shape):
        """Build the connection."""
        input_dim = int(input_shape[1])  # the real input dim
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"set_input_dim was called with {self.input_dim}, "
                f"but acutal input has dimension {input_dim}"
            )
        if self.input_dim is None:
            # invoke the function to set_input_dim
            self.set_input_dim(input_dim)

    # set the dimension of the input of NCP,
    # also create the matrix sensory_adjaceny_matrix,
    def set_input_dim(self, input_dim):
        """Set the input dimension."""
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32)
        # (sensory neurons * hidden neurons)
        # save the synapses polarity

    # set the output_dim of the neuron,
    # in our case should be 3:(steering, throttle, brake)
    def set_output_dim(self, output_dim):
        """Set the output dimension."""
        self.output_dim = output_dim

    def erev_initializer(self):
        """Initialize the E_ij for layers except sensory layer."""
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self):
        """Initialize the E_ij for sensory layers."""
        return np.copy(self.sensory_adjacency_matrix)

    def add_synapse(self, src, dest, polarity):
        """Set the synapse between hidden neurons."""
        # and synapse between sensory neurons and hidden neurons
        if src < 0 or src >= self.units:
            raise ValueError(
                f"cannot add synapse originating in {src}, "
                f"if cell has only {self.units} units.")
        if dest < 0 or dest >= self.units:
            raise ValueError(
                f"cannot add synapse feeding into {dest}, "
                f"if cell has only {self.units} units"
            )
        if polarity not in [-1, 1]:
            raise ValueError(f"cannot add synapse with polarity {polarity} "
                             "expected(-1 or +1)")
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        """Add synapse connection between sensory layers and inter layers."""
        if self.input_dim is None:
            raise ValueError("cannot add sensory_synapses "
                             "before build() has been called")
        if src < 0 or src >= self.input_dim:
            raise ValueError(f"cannot add sensory synapse "
                             f"originating in {src},"
                             f"if cell has only {self.input_dim} features")
        if dest < 0 or dest >= self.units:
            raise ValueError(f"cannot add synapse feeding into {dest}, "
                             f"if cell has only {self.units} units")
        if polarity not in [-1, 1]:
            raise ValueError(f"cannot add synapse with polarity {polarity} "
                             "expected(-1 or +1)")
        self.sensory_adjacency_matrix[src, dest] = polarity

    def get_config(self):
        """Get the configuration how the neurons are wired."""
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "units": self.units
        }


class NCPWiring(Wiring):
    """This is the class of how NCP neurons are wired."""

    def __init__(self, inter_neurons, command_neurons, motor_neurons,
                 sensory_fanout, inter_fanout, recurrent_command,
                 motor_fanin, seed=22222):
        """Initialize the object."""
        super(NCPWiring,
              self).__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)  # create a random generator
        self._num_inter_neurons = inter_neurons
        self._num_command_neurons = command_neurons
        self._num_motor_neurons = motor_neurons
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command = recurrent_command
        self._motor_fanin = motor_fanin
        # initialize the following 2 variables in Init, give them value in
        # method build()
        self._num_sensory_neurons = None
        self._sensory_neurons = None

        # neuron IDs: [0...motor... command...inter]
        self._motor_neurons = list(range(0, self._num_motor_neurons))
        self._command_neurons = list(range(
            self._num_motor_neurons,
            self._num_motor_neurons+self._num_command_neurons)
        )
        self._inter_neurons = list(range(
            self._num_motor_neurons + self._num_command_neurons,
            self._num_motor_neurons + self._num_command_neurons +
            self._num_inter_neurons,
        ))

        # check if the settings of motor_fanin, sensory_fanout,
        # inter_fanout are reasonable
        if self._sensory_fanout > self._num_inter_neurons:
            raise ValueError(f"Error: Sensory_fanout is {self._sensory_fanout}"
                             f" but there are only {self._num_inter_neurons}"
                             f" inter neurons")

        if self._inter_fanout > self._num_command_neurons:
            raise ValueError(f"Error: inter_fanout is {self._inter_fanout} but"
                             f" there are only {self._num_command_neurons}"
                             f" command neurons")

        if self._motor_fanin > self._num_command_neurons:
            raise ValueError(f"Error: motor_fanin is {self._motor_fanin} but"
                             f" there are only {self._num_command_neurons}"
                             f" command neurons")

    def get_type_of_neuron(self, neuron_id):
        """Get the neuron type."""
        if neuron_id < self._num_motor_neurons:
            return "motor"
        if neuron_id < self._num_motor_neurons + self._num_command_neurons:
            return "command"
        return "inter"

    def build_sensory_to_inter(self):
        """Connect the sensory neurons to inter-neurons."""
        # all the inter_neurons are unreachable in the beginning
        # unreachable_inter_neurons = [neuron for
        #                             neuron in self._inter_neurons]
        unreachable_inter_neurons = list(self._inter_neurons)
        # randomly connect each sensory neuron to the
        # _sensory_fanout number of interneurons
        for src in self._sensory_neurons:
            for dest in self._rng.choice(
                    self._inter_neurons,
                    size=self._sensory_fanout, replace=False):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)
        # some inter neurons are not connected
        mean_inter_neuron_fanin = int(
            self._num_sensory_neurons *
            self._sensory_fanout / self._num_inter_neurons)
        for dest in unreachable_inter_neurons:
            for src in self._rng.choice(
                    self._sensory_neurons,
                    size=mean_inter_neuron_fanin, replace=False):
                polarity = self._rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def build_inter_to_command(self):
        """Connect the inter-neurons to command-neurons."""
        # unreachable_command_neurons = [neuron for
        #                               neuron in self._command_neurons]
        unreachable_command_neurons = list(self._command_neurons)
        for src in self._inter_neurons:
            for dest in self._rng.choice(
                    self._command_neurons,
                    size=self._inter_fanout, replace=False):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        mean_command_neuron_fanin = int(
            self._inter_fanout *
            self._num_inter_neurons/self._num_command_neurons)
        for dest in unreachable_command_neurons:
            for src in self._rng.choice(
                    self._inter_neurons,
                    size=mean_command_neuron_fanin, replace=False):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build_recurrent_command(self):
        """Build the recurrence among command-neurons."""
        for _ in range(self._recurrent_command):
            # print(f"recurrence connection {i}.")
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            polarity = self._rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def build_command_to_motor(self):
        """Connect the command-neuron to motor neuron."""
        # unreachable_command_neurons = [neuron for
        #                               neuron in self._command_neurons]
        unreachable_command_neurons = list(self._command_neurons)
        for dest in self._motor_neurons:
            for src in self._rng.choice(
                    self._command_neurons,
                    size=self._motor_fanin, replace=False):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        mean_command_fanout = int(
            self._num_motor_neurons *
            self._motor_fanin / self._num_command_neurons)

        for src in unreachable_command_neurons:
            for dest in self._rng.choice(
                    self._motor_neurons,
                    size=mean_command_fanout,
                    replace=False):
                polarity = self._rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        """Build the connection."""
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(0, self._num_sensory_neurons))
        self.build_sensory_to_inter()
        self.build_inter_to_command()
        self.build_recurrent_command()
        self.build_command_to_motor()

    def get_config(self):
        """Get configuration of NCP wiring."""
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "sensory_fanout": self._sensory_fanout,
            "inter_neuron":  self._num_inter_neurons,
            "inter_fanout": self._inter_fanout,
            "command_neuron": self._num_command_neurons,
            "command_recurrency": self._recurrent_command,
            "output_dim": self.output_dim,
            "motor_fanin": self._motor_fanin,
            "units": self.units
        }
