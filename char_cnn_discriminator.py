import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import os, re, sys
from jexus import Clock
global device
device = "cuda:0"


DIM = 512
SEQ_LEN = 15 + 2
WORD_DIM = 1024

class Resblock(nn.Module):
    def __init__(self, inner_dim, kernel_size):
        super(Resblock, self).__init__()
        self.inner_dim = inner_dim
        self.kernel_size = kernel_size
        self.relu = nn.ReLU()
        if kernel_size % 2 != 1:
            raise Exception("kernel size must be odd number!")
        self.conv_1 = nn.Conv1d(self.inner_dim, self.inner_dim, self.kernel_size, padding=int((kernel_size-1)/2))
        self.conv_2 = nn.Conv1d(self.inner_dim, self.inner_dim, self.kernel_size, padding=int((kernel_size-1)/2))

    def forward(self, inputs):
        output = self.relu(inputs)
        output = self.conv_1(output)
        output = self.relu(output)
        output = self.conv_2(output)
        return inputs + (0.3*output)


class Discriminator(nn.Module):
    def __init__(self, word_dim, inner_dim, seq_len, kernel_size=3, device="cuda:0"):
        super(Discriminator, self).__init__()
        self.device = device
        self.word_dim = word_dim
        self.inner_dim = inner_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        if kernel_size % 2 != 1:
            raise Exception("kernel size must be odd number!")
        self.conv_1 = nn.Conv1d(self.word_dim, self.inner_dim, self.kernel_size, padding=int((kernel_size-1)/2))
        self.resblock_1 = Resblock(inner_dim, kernel_size)
        self.resblock_2 = Resblock(inner_dim, kernel_size)
        self.resblock_3 = Resblock(inner_dim, kernel_size)
        self.resblock_4 = Resblock(inner_dim, kernel_size)
        W = seq_len*inner_dim
        self.fc_1 = nn.Linear(W, int(W/8))
        self.fc_2 = nn.Linear(int(W/8), int(W/32))
        self.fc_3 = nn.Linear(int(W/32), int(W/64))
        self.fc_4 = nn.Linear(int(W / 64), 2)
        self.relu = nn.LeakyReLU()
        
    def feed_fc(self, inputs):
        output = self.relu(self.fc_1(inputs))
        output = self.relu(self.fc_2(output))
        output = self.relu(self.fc_3(output))
        return self.fc_4(output)

    def forward(self, inputs):
        this_bs = inputs.shape[0]
        inputs = inputs.permute(0, 2, 1).float()
        if inputs.shape[-1] != self.seq_len:
            # print("Warning: seq_len(%d) != fixed_seq_len(%d), auto-pad."%(inputs.shape[-1], self.seq_len))
            p1d = (0, self.seq_len - inputs.shape[-1])
            inputs = F.pad(inputs, p1d, "constant", 0)
            # print("after padding,", inputs.shape)
        output = self.conv_1(inputs)
        output = self.resblock_1(output)
        output = self.resblock_2(output)
        output = self.resblock_3(output)
        output = self.resblock_4(output)
        output = output.view(this_bs, -1)
        # print(output.shape)
        return self.feed_fc(output)
            



# def ResBlock(name, inputs):
#     output = inputs
#     output = tf.nn.relu(output)
#     output = tflib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 3, output)
#     output = tf.nn.relu(output)
#     output = tflib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 3, output)
#     return inputs + (0.3*output)


# def discriminator_X(inputs):
#     output = tf.transpose(inputs, [0,2,1])
#     output = tflib.ops.conv1d.Conv1D('discriminator_x.Input',WORD_DIM, DIM, 1, output)
#     output = ResBlock('discriminator_x.1', output)
#     output = ResBlock('discriminator_x.2', output)
#     output = ResBlock('discriminator_x.3', output)
#     output = ResBlock('discriminator_x.4', output)
#     #output = ResBlock('Discriminator.5', output)

#     output = tf.reshape(output, [-1, SEQ_LEN*DIM])
#     output = tflib.ops.linear.Linear('discriminator_x.Output', SEQ_LEN*DIM, 1, output)
#     return tf.squeeze(output,[1])