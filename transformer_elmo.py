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
from transformer import MultiHeadedAttention, PositionwiseFeedForward, \
                        PositionalEncoding, EncoderDecoder, \
                        Encoder, EncoderLayer, Decoder, DecoderLayer, \
                        subsequent_mask

from Embedder import *

def make_model_elmo(N=6, d_model=1024, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embedder(), c(position)),
        nn.Sequential(Embedder(), c(position)),
        generator=None)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Batch():
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, max_len=40):
        self.src = src
        length = [min(max_len, len(x))+2 for x in self.src]
        self.src_mask = torch.zeros((len(length), max_len + 2))
        self.max_len = max_len
        for i,j in enumerate(length):
            self.src_mask[i,range(j)]=1
        
        if trg is not None:
            self.trg = trg
            # self.trg_y = trg
            self.trg_mask = \
                self.make_std_mask(self.trg, max_len)
            self.ntokens = self.src_mask.data.sum()
    
    @staticmethod
    def make_std_mask(tgt, max_len):
        "Create a mask to hide padding and future words."
        length = [min(max_len, len(x))+1 for x in tgt]
        tgt_mask = torch.zeros((len(length), max_len + 1))
        for i,j in enumerate(length):
            tgt_mask[i,range(j)]=1
        # tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(max_len + 1).type_as(tgt_mask.data))
        return tgt_mask

def pretrain_data_gen(iterator):
    "Generate random data for a src-tgt copy task."
    for i in iterator:
        yield Batch(i, i)

def pretrain_run_epoch(data_iter, model, loss_compute, train_step_num):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ct = Clock(train_step_num)
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        ct.flush(info={"loss":loss / batch.ntokens.float().to(device)})
    return total_loss / total_tokens.float().to(device)