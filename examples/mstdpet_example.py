import torch
import matplotlib.pyplot as plt

from pysnn.network import SpikingModule
from pysnn.connection import Linear
from pysnn.neuron import LIFNeuron
from pysnn.learning import MSTDPET


#########################################################
# Params
#########################################################
# Architecture
inputs = 1
outputs = 1
c_shape = (inputs, outputs)

# Data
batch_size = 1

# Neuronal dynamics
dt = 1
alpha_v = 0.3
alpha_t = 1.0
thresh = 0.5
v_rest = 0.0
duration_refrac = 3
voltage_decay = 0.8
trace_decay = 0.8
delay = 0
n_dynamics = (
    thresh,
    v_rest,
    alpha_v,
    alpha_t,
    dt,
    duration_refrac,
    voltage_decay,
    trace_decay,
)
n2_dynamics = (thresh * 2, *n_dynamics[1:])
n_in_dynamics = (dt, alpha_t, trace_decay)
c_dynamics = (batch_size, dt, delay)

# Learning
a_pre = 1.0
a_post = 1.0
lr = 0.1
e_trace_decay = 0.8
l_params = (lr, a_pre, a_post, e_trace_decay)


#########################################################
# Network
#########################################################
class Network(SpikingModule):
    def __init__(self):
        super(Network, self).__init__()

        # One layer
        self.pre_neuron = LIFNeuron((batch_size, 1, inputs), *n_dynamics)
        self.post_neuron = LIFNeuron((batch_size, 1, outputs), *n2_dynamics)
        self.linear = Linear(*c_shape, *c_dynamics)

    def forward(self, input):
        x, t = self.pre_neuron(input)
        x, t = self.linear(x, t)
        x, t = self.post_neuron(x)

        return x, t

    def reset_state(self):
        for module in self.spiking_children():
            module.reset_state()


#########################################################
# Training
#########################################################
device = torch.device("cpu")
net = Network()
layers, _, _, _ = net.trace_graph(torch.zeros((batch_size, 1, inputs)))
net.linear.reset_weights("constant", 3)
learning_rule = MSTDPET(layers, *l_params)

# Training loop
current = []
pre_spikes = []
post_spikes = []
pre_trace = []
post_trace = []
rewards = []
e_trace = []
weight = []

for i in range(100):
    # Generate input spikes
    curr = (torch.rand(batch_size, 1, inputs) > 0.4).float()
    current.append(curr.item())

    # Do forward pass
    post_s, post_t = net.forward(curr)

    # Append network spikes and traces
    pre_spikes.append(net.pre_neuron.spikes.item())
    post_spikes.append(post_s.item())
    pre_trace.append(net.pre_neuron.trace.item())
    post_trace.append(post_t.item())

    # Do learning pass
    if i < 50:
        reward = 1.0
    else:
        reward = -1.0
    learning_rule.step(reward)

    # Append last resulting items
    rewards.append(reward)
    e_trace.append(net.linear.e_trace.item())
    weight.append(net.linear.weight.item())

# Reset states
net.reset_state()
learning_rule.reset_state()

# Plotting network state and spikes
fig, axs = plt.subplots(8, 1)

for ax, data in zip(
    axs,
    [
        current,
        pre_spikes,
        post_spikes,
        pre_trace,
        post_trace,
        rewards,
        e_trace,
        weight,
    ],
):
    ax.plot(data)

plt.show()
