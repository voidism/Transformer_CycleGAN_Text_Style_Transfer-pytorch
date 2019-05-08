from attn import Transformer, LabelSmoothing, \
data_gen, NoamOpt, Generator, SimpleLossCompute, \
greedy_decode, subsequent_mask, TransformerEncoder
import torch
import torch.nn.functional as F
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, word_dim, inner_dim, seq_len, N=3, device="cuda:0", kernel_size=3):
        super(Discriminator, self).__init__()
        self.encoder = TransformerEncoder(N, word_dim, inner_dim)
        self.seq_len = seq_len
        self.conv_1 = nn.Conv1d(word_dim, inner_dim, kernel_size, padding=int((kernel_size-1)/2))
        self.conv_2 = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=int((kernel_size-1)/2))
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

    def forward(self, src, src_mask=None):
        this_bs = src.shape[0]
        x = self.encoder(src, src_mask)
        inputs = x.permute(0, 2, 1).float()
        if inputs.shape[-1] != self.seq_len:
            # print("Warning: seq_len(%d) != fixed_seq_len(%d), auto-pad."%(inputs.shape[-1], self.seq_len))
            p1d = (0, self.seq_len - inputs.shape[-1])
            inputs = F.pad(inputs, p1d, "constant", 0)
            # print("after padding,", inputs.shape)
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = x.view(this_bs, -1)
        return self.feed_fc(x)