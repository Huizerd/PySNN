"""NOTE: This example does not work yet!"""

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from pysnn.network import SpikingModule
from pysnn.connection import Conv2d, AdaptiveMaxPool2d
from pysnn.neuron import LIFNeuron, Input
from pysnn.learning import MSTDPET
from pysnn.utils import conv2d_output_shape
from pysnn.datasets import nmnist_train_test


#########################################################
# Params
#########################################################
# Architecture
n_in = 5
n_hidden1 = 10
n_hidden2 = 10
n_out = 5

# Data
sample_length = 300
num_workers = 0
batch_size = 2

# Neuronal dynamics
thresh = 0.8
v_rest = 0
alpha_v = 0.2
tau_v = 5
alpha_t = 1.0
tau_t = 5
duration_refrac = 5
dt = 1
delay = 3
n_dynamics = (
    thresh,
    v_rest,
    alpha_v,
    alpha_v,
    dt,
    duration_refrac,
    tau_v,
    tau_t,
    "exponential",
)
c_dynamics = (batch_size, dt, delay)
i_dynamics = (dt, alpha_t, tau_t, "exponential")

# Learning
lr = 0.0001


#########################################################
# Network
#########################################################
class Network(SpikingModule):
    def __init__(self):
        super(Network, self).__init__()

        # Input
        self.input = Input((batch_size, 2, 34, 34), *i_dynamics)

        # Layer 1
        self.conv1 = Conv2d(2, 4, 5, (34, 34), *c_dynamics, padding=1, stride=1)
        self.neuron1 = LIFNeuron((batch_size, 4, 32, 32), *n_dynamics)

        # Layer 2
        # self.pool2 = AdaptiveMaxPool2d((16, 16))
        self.conv2 = Conv2d(4, 8, 5, (32, 32), *c_dynamics, padding=1, stride=2)
        self.neuron2 = LIFNeuron((batch_size, 8, 15, 15), *n_dynamics)

        # Layer out
        self.conv3 = Conv2d(8, 1, 3, (15, 15), *c_dynamics)
        self.neuron3 = LIFNeuron((batch_size, 1, 13, 13), *n_dynamics)

    def forward(self, input):
        x, t = self.input(input)

        # Layer 1
        x, t = self.conv1(x, t)
        x, t = self.neuron1(x)

        # Layer 2
        # x = self.pool2(x)
        x, t = self.conv2(x, t)
        x, t = self.neuron2(x)

        # Layer out
        x, t = self.conv3(x, t)
        x, t = self.neuron3(x)

        return x, t

    def reset_state(self):
        for module in self.spiking_children():
            module.reset_state()


#########################################################
# Dataset
#########################################################
root = "nmnist"
if os.path.isdir(root):
    train_dataset, test_dataset = nmnist_train_test(root)
else:
    raise NotADirectoryError(
        "Make sure to download the N-MNIST dataset from https://www.garrickorchard.com/datasets/n-mnist and put it in the 'nmnist' folder."
    )

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


#########################################################
# Training
#########################################################
device = torch.device("cpu")
net = Network()
layers, _, _, _ = net.trace_graph(next(iter(train_dataloader))[0][..., 0])
learning_rule = MSTDPET(layers, lr)

print("Input image shape: ", next(iter(train_dataloader))[0].shape)

print("Convolution connection 1 weight shape: ", net.conv1.weight.shape)
print("Convolution connection 1 trace shape: ", net.conv1.trace.shape)
print("Neuron 1 trace shape: ", net.neuron1.trace.shape)

print("Convolution connection 2 weight shape: ", net.conv2.weight.shape)
print("Convolution connection 2 trace shape: ", net.conv2.trace.shape)
print("Neuron 2 trace shape: ", net.neuron2.trace.shape)

print("Convolution connection 3 weight shape: ", net.conv3.weight.shape)
print("Convolution connection 3 trace shape: ", net.conv3.trace.shape)
print("Neuron 3 trace shape: ", net.neuron3.trace.shape)

output = []
for batch in tqdm(train_dataloader):
    input = batch[0]
    for idx in range(input.shape[-1]):
        x = input[..., idx].to(device)
        out, _ = net(x)
        output.append(out)

        learning_rule.step(1)

    net.reset_state()

    # Break because this would take way too long
    break

print(torch.stack(output, dim=-1).shape)
