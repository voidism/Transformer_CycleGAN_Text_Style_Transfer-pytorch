import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import os, re, sys
from jexus import Clock
from attn import Transformer, LabelSmoothing, \
data_gen, NoamOpt, Generator, SimpleLossCompute, \
greedy_decode, subsequent_mask
from utils import Utils
from char_cnn_discriminator import Discriminator
import argparse

device = "cuda:1"


def prob_backward(model, embed, src, src_mask, max_len, start_symbol=2, raw=False):
    if raw==False:
        memory = model.encode(embed(src.to(device)), src_mask)
    else:
        memory = model.encode(src.to(device), src_mask)

    ys = torch.ones(src.shape[0], 1, dtype=torch.int64).fill_(start_symbol).to(device)
    probs = []
    for i in range(max_len+2-1):
        out = model.decode(memory, src_mask, 
                        embed(Variable(ys)), 
                        Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        probs.append(prob.unsqueeze(1))
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    ret = torch.cat(probs, dim=1)
    return ret

def perplexity(prob):
    entropy = -(prob * torch.log(prob)).sum(-1)
    return torch.exp(entropy.mean())


def backward_decode(model, embed, src, src_mask, max_len, start_symbol=2, raw=False, return_term=-1):
    if raw==False:
        memory = model.encode(embed(src.to(device)), src_mask)
    else:
        memory = model.encode(src.to(device), src_mask)

    ys = torch.ones(src.shape[0], 1, dtype=torch.int64).fill_(start_symbol).to(device)
    ret_back = embed(ys).float()
    if return_term == 2:
        ppls = 0
    for i in range(max_len+2-1):
        out = model.decode(memory, src_mask, 
                        embed(Variable(ys)), 
                        Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator.scaled_forward(out[:, -1], scale=10.0)
        if return_term == 2:
            ppls += perplexity(model.generator.scaled_forward(out[:, -1], scale=1.0))
        back = torch.matmul(prob ,embed.weight.data.float())
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        ret_back = torch.cat([ret_back, back.unsqueeze(1)], dim=1)
    return (ret_back, ys) if return_term == -1 else ret_back if return_term == 0 else ys if return_term == 1 else (ret_back, ppls) if return_term == 2 else None

def reconstruct(model, src, max_len, start_symbol=2):
    memory = model.encoder(model.src_embed[1](src), None)
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).long().to(device)
    ret_back = model.tgt_embed[0].pure_emb(ys).float()
    for i in range(max_len-1):
        out = model.decode(memory, None, 
                        Variable(ys), 
                        Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        back = torch.matmul(prob ,model.tgt_embed[0].lut.weight.data.float())
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        ret_back = torch.cat([ret_back, back.unsqueeze(1)], dim=1)
    return ret_back

def wgan_pg(netD, fake_data, real_data, lamb=10):
    batch_size = fake_data.shape[0]
    ## 1. interpolation
    alpha = torch.rand(batch_size, 1, 1).expand(real_data.size()).to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates.to(device), requires_grad=True)
    ## 2. gradient penalty
    disc_interpolates = netD(interpolates).view(batch_size, )
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    ## 3. append it to loss function
    return gradient_penalty

class CycleGAN(nn.Module):
    def __init__(self, discriminator, generator, utils, embedder):
        super(CycleGAN, self).__init__()
        self.D = discriminator
        self.G = generator
        self.R = copy.deepcopy(generator)
        self.D_opt = torch.optim.Adam(self.D.parameters())
        # self.G_opt = torch.optim.Adam(self.G.parameters())
        self.G_opt = NoamOpt(utils.emb_mat.shape[1], 1, 4000,
            torch.optim.Adam(self.G.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        # self.R_opt = torch.optim.Adam(self.R.parameters())
        self.R_opt = NoamOpt(utils.emb_mat.shape[1], 1, 4000,
            torch.optim.Adam(self.R.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.embed = embedder

        self.utils = utils
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.cosloss=nn.CosineEmbeddingLoss()
        self.r_criterion = LabelSmoothing(size=utils.emb_mat.shape[0], padding_idx=0, smoothing=0.0)
        self.r_loss_compute = SimpleLossCompute(self.R.generator, self.r_criterion, self.R_opt)

    def save_model(self, d_path="Dis_model.ckpt", g_path="Gen_model.ckpt", r_path="Res_model.ckpt"):
        torch.save(self.D.state_dict(), d_path)
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.R.state_dict(), r_path)

    def load_model(self, path="", g_file=None, d_file=None, r_file=None):
        if g_file!=None:
            self.G.load_state_dict(torch.load(os.path.join(path, g_file), map_location=device))
        if d_file!=None:
            self.D.load_state_dict(torch.load(os.path.join(path, d_file), map_location=device))
        if r_file!=None:
            self.R.load_state_dict(torch.load(os.path.join(path, r_file), map_location=device))
        print("model loaded!")

    def pretrain_disc(self, num_epochs=100):
        X_datagen = self.utils.data_generator("X")
        Y_datagen = self.utils.data_generator("Y")
        for epoch in range(num_epochs):
            d_steps = self.utils.train_step_num
            d_ct = Clock(d_steps, title="Train Discriminator(%d/%d)"%(epoch, num_epochs))
            for step, X_data, Y_data in zip(range(d_steps), data_gen(X_datagen, self.utils.sents2idx), data_gen(Y_datagen, self.utils.sents2idx)):
                # 1. Train D on real+fake
                # if epoch == 0:
                #     break
                self.D.zero_grad()
        
                #  1A: Train D on real
                d_real_pred = self.D(self.embed(Y_data.src.to(device)))
                d_real_error = self.criterion(d_real_pred, torch.ones((d_real_pred.shape[0],), dtype=torch.int64).to(device))  # ones = true

                #  1B: Train D on fake
                d_fake_pred = self.D(self.embed(X_data.src.to(device)))
                d_fake_error = self.criterion(d_fake_pred, torch.zeros((d_fake_pred.shape[0],), dtype=torch.int64).to(device))  # zeros = fake
                (d_fake_error + d_real_error).backward()
                self.D_opt.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
                d_ct.flush(info={"D_loss": d_fake_error.item()})
        torch.save(self.D.state_dict(), "model_disc_pretrain.ckpt")

    def train_model(self, num_epochs=100, g_scale=1.0, r_scale=1.0):
        for i, batch in enumerate(data_gen(utils.test_generator("X"), utils.sents2idx)):
            X_test_batch = batch
            break

        for i, batch in enumerate(data_gen(utils.test_generator("Y"), utils.sents2idx)):
            Y_test_batch = batch
            break
        X_datagen = self.utils.data_generator("X")
        Y_datagen = self.utils.data_generator("Y")
        for epoch in range(num_epochs):
            ct = Clock(self.utils.train_step_num, title="Train G/D (%d/%d)" % (epoch, num_epochs))
            for i, X_data, Y_data in zip(range(self.utils.train_step_num), data_gen(X_datagen, self.utils.sents2idx), data_gen(Y_datagen, self.utils.sents2idx)):
                for d_step in range(4):
                    # 1. Train D on real+fake
                    # if epoch == 0:
                    #     break
                    self.D.zero_grad()
            
                    #  1A: Train D on real
                    d_real_data = self.embed(Y_data.src.to(device)).float()
                    d_real_pred = self.D(d_real_data)
                    # d_real_error = self.criterion(d_real_pred, torch.ones((d_real_pred.shape[0],), dtype=torch.int64).to(device))  # ones = true

                    #  1B: Train D on fake
                    self.G.to(device)
                    d_fake_data = backward_decode(self.G, self.embed, X_data.src, X_data.src_mask, max_len=self.utils.max_len, return_term=0).detach()  # detach to avoid training G on these labels
                    d_fake_pred = self.D(d_fake_data)
                    # d_fake_error = self.criterion(d_fake_pred, torch.zeros((d_fake_pred.shape[0],), dtype=torch.int64).to(device))  # zeros = fake
                    # (d_fake_error + d_real_error).backward()
                    d_loss = d_fake_pred.mean() - d_real_pred.mean()
                    d_loss += wgan_pg(self.D, d_fake_data, d_real_data, lamb=10)

                    d_loss.backward()
                    self.D_opt.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

                self.G.zero_grad()
                g_fake_data, g_ppl = backward_decode(self.G, self.embed, X_data.src, X_data.src_mask, max_len=self.utils.max_len, return_term=2)
                dg_fake_pred = self.D(g_fake_data)
                # g_error = self.criterion(dg_fake_pred, torch.ones((dg_fake_pred.shape[0],), dtype=torch.int64).to(device))  # we want to fool, so pretend it's all genuine
                g_loss = -dg_fake_pred.mean() + 0.1*(60.0 - g_ppl)**2

                g_loss.backward(retain_graph=True)
                self.G_opt.step()  # Only optimizes G's parameters
                self.G.zero_grad()

                # 3. reconstructor  643988636173-69t5i8ehelccbq85o3esu11jgh61j8u5.apps.googleusercontent.com
                # way_3
                out = self.R.forward(g_fake_data, embedding_layer(X_data.trg.to(device)), 
                    None, X_data.trg_mask)
                r_loss = 10000*self.r_loss_compute(out, X_data.trg_y, X_data.ntokens)
                self.G_opt.step()
                self.G_opt.optimizer.zero_grad()
                ct.flush(info={"D": d_loss.item(),
                "G": g_loss.item(),
                "R": r_loss / X_data.ntokens.float().to(device)})
                    # way_2
                    # r_reco_data = prob_backward(self.R, self.embed, g_fake_data, None, max_len=self.utils.max_len, raw=True)
                    # x_orgi_data = X_data.src[:, 1:]
                    # r_loss = SimpleLossCompute(None, criterion, self.R_opt)(r_reco_data, x_orgi_data, X_data.ntokens)
                    # way_1
                    # viewed_num = r_reco_data.shape[0]*r_reco_data.shape[1]
                    # r_error = r_scale*self.cosloss(r_reco_data.float().view(-1, self.embed.weight.shape[1]), x_orgi_data.float().view(-1, self.embed.weight.shape[1]), torch.ones(viewed_num, dtype=torch.float32).to(device))
                if i%100 == 0:
                    with torch.no_grad():
                        x_cont, x_ys = backward_decode(model, self.embed, X_test_batch.src, X_test_batch.src_mask, max_len=25, start_symbol=2)
                        x = utils.idx2sent(x_ys)
                        y_cont, y_ys = backward_decode(model, self.embed, Y_test_batch.src, Y_test_batch.src_mask, max_len=25, start_symbol=2)
                        y = utils.idx2sent(y_ys)
                        r_x = utils.idx2sent(backward_decode(self.R, self.embed, x_cont, None, max_len=self.utils.max_len, raw=True, return_term=1))
                        r_y = utils.idx2sent(backward_decode(self.R, self.embed, y_cont, None, max_len=self.utils.max_len, raw=True, return_term=1))

                        for i,j,l in zip(X_test_batch.src, x, r_x):
                            print("===")
                            k = utils.idx2sent([i])[0]
                            print("ORG:", " ".join(k[:k.index('<eos>')+1]))
                            print("GEN:", " ".join(j[:j.index('<eos>')+1] if '<eos>' in j else j))
                            print("REC:", " ".join(l[:l.index('<eos>')+1] if '<eos>' in l else l))
                        print("=====")
                        for i, j, l in zip(Y_test_batch.src, y, r_y):
                            print("===")
                            k = utils.idx2sent([i])[0]
                            print("ORG:", " ".join(k[:k.index('<eos>')+1]))
                            print("GEN:", " ".join(j[:j.index('<eos>')+1] if '<eos>' in j else j))
                            print("REC:", " ".join(l[:l.index('<eos>')+1] if '<eos>' in l else l))
                    self.save_model()

def pretrain_run_epoch(data_iter, model, loss_compute, train_step_num, embedding_layer):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    ct = Clock(train_step_num)
    embedding_layer.to(device)
    model.to(device)
    for i, batch in enumerate(data_iter):
        batch.to(device)
        out = model.forward(embedding_layer(batch.src.to(device)), embedding_layer(batch.trg.to(device)), 
                batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        batch.to("cpu")
        if i % 50 == 1:
            elapsed = time.time() - start
            ct.flush(info={"loss":loss / batch.ntokens.float().to(device), "tok/sec":tokens.float().to(device) / elapsed})
            # print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            #         (i, loss / batch.ntokens.float().to(device), tokens.float().to(device) / elapsed))
            start = time.time()
            tokens = 0
        else:
            ct.flush(info={"loss":loss / batch.ntokens.float().to(device)})
    return total_loss / total_tokens.float().to(device)

def get_embedding_layer(utils):
    d_model = utils.emb_mat.shape[1]
    vocab = utils.emb_mat.shape[0]
    embedding_layer =  nn.Embedding(vocab, d_model)
    embedding_layer.weight.data = torch.tensor(utils.emb_mat)
    embedding_layer.weight.requires_grad = False
    embedding_layer.to(device)
    return embedding_layer

def pretrain(model, embedding_layer, utils, epoch_num=1):
    criterion = LabelSmoothing(size=utils.emb_mat.shape[0], padding_idx=0, smoothing=0.0)
    model_opt = NoamOpt(utils.emb_mat.shape[1], 1, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    X_test_batch = None
    Y_test_batch = None
    for i, batch in enumerate(data_gen(utils.data_generator("X"), utils.sents2idx)):
        X_test_batch = batch
        break

    for i, batch in enumerate(data_gen(utils.data_generator("Y"), utils.sents2idx)):
        Y_test_batch = batch
        break
    model.to(device)
    for epoch in range(epoch_num):
            model.train()
            print("EPOCH %d:"%(epoch+1))
            pretrain_run_epoch(data_gen(utils.data_generator("Y"), utils.sents2idx), model, 
                    SimpleLossCompute(model.generator, criterion, model_opt), utils.train_step_num, embedding_layer)
            pretrain_run_epoch(data_gen(utils.data_generator("X"), utils.sents2idx), model, 
                    SimpleLossCompute(model.generator, criterion, model_opt), utils.train_step_num, embedding_layer)
            model.eval()
            torch.save(model.state_dict(), 'model_pretrain.ckpt')
            x = utils.idx2sent(greedy_decode(model, embedding_layer, X_test_batch.src, X_test_batch.src_mask, max_len=20, start_symbol=2))
            y = utils.idx2sent(greedy_decode(model, embedding_layer, Y_test_batch.src, Y_test_batch.src_mask, max_len=20, start_symbol=2))

            for i,j in zip(X_test_batch.src, x):
                print("===")
                k = utils.idx2sent([i])[0]
                print("ORG:", " ".join(k[:k.index('<eos>')+1]))
                print("--")
                print("GEN:", " ".join(j[:j.index('<eos>')+1] if '<eos>' in j else j))
                print("===")
            print("=====")
            for i, j in zip(Y_test_batch.src, y):
                print("===")
                k = utils.idx2sent([i])[0]
                print("ORG:", " ".join(k[:k.index('<eos>')+1]))
                print("--")
                print("GEN:", " ".join(j[:j.index('<eos>')+1] if '<eos>' in j else j))
                print("===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="execute mode")
    parser.add_argument("-filename", default=None, required=False, help="test filename")
    parser.add_argument("-load_model", default=False, required=False, help="test filename")
    parser.add_argument("-model_name", default="model.ckpt", required=False, help="test filename")
    parser.add_argument("-disc_name", default="cnn_disc/model_disc_pretrain_xy_inv.ckpt", required=False, help="test filename")
    parser.add_argument("-save_path", default="", required=False, help="test filename")
    parser.add_argument("-X_file", default="shuf_cna.txt", required=False, help="X domain text filename")
    parser.add_argument("-Y_file", default="shuf_cou.txt", required=False, help="Y domain text filename")
    parser.add_argument("-X_test", default="test.cna", required=False, help="X domain text filename")
    parser.add_argument("-Y_test", default="test.cou", required=False, help="Y domain text filename")
    parser.add_argument("-epoch", default=1, required=False, help="test filename")
    parser.add_argument("-max_len", default=20, required=False, help="test filename")
    parser.add_argument("-batch_size", default=32, required=False, help="batch size")
    args = parser.parse_args()

    model = Transformer(N=2)
    utils = Utils(X_data_path=args.X_file, Y_data_path=args.Y_file,
    X_test_path=args.X_test, Y_test_path=args.Y_test,
    batch_size=int(args.batch_size))
    embedding_layer = get_embedding_layer(utils).to(device)
    model.generator = Generator(d_model = utils.emb_mat.shape[1], vocab=utils.emb_mat.shape[0])
    if args.load_model:
        model.load_state_dict(torch.load(args.model_name))
    if args.mode == "pretrain":
        pretrain(model, embedding_layer, utils, int(args.epoch))
    if args.mode == "cycle":
        disc = Discriminator(word_dim=utils.emb_mat.shape[1], inner_dim=512, seq_len=20)
        main_model = CycleGAN(disc, model, utils, embedding_layer)
        main_model.to(device)
        main_model.load_model(g_file="model_pretrain.ckpt", r_file="model_pretrain.ckpt", d_file=None)
        main_model.train_model()
    if args.mode == "disc":
        disc = Discriminator(word_dim=utils.emb_mat.shape[1], inner_dim=512, seq_len=20)
        main_model = CycleGAN(disc, model, utils, embedding_layer)
        main_model.to(device)
        main_model.pretrain_disc(2)

    if args.mode == "dev":
        model = Transformer(N=2)
        utils = Utils(X_data_path="big_cou.txt", Y_data_path="big_cna.txt")
        model.generator = Generator(d_model = utils.emb_mat.shape[1], vocab=utils.emb_mat.shape[0])
        criterion = LabelSmoothing(size=utils.emb_mat.shape[0], padding_idx=0, smoothing=0.0)
        model_opt = NoamOpt(utils.emb_mat.shape[1], 1, 400,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        X_test_batch = None
        Y_test_batch = None
        for i, batch in enumerate(data_gen(utils.data_generator("X"), utils.sents2idx)):
            X_test_batch = batch
            break

        for i, batch in enumerate(data_gen(utils.data_generator("Y"), utils.sents2idx)):
            Y_test_batch = batch
            break

    # if args.load_model:
    #     model.load_model(filename=args.model_name)
    # if args.mode == "train":
    #     model.train_model(num_epochs=int(args.epoch))
    #     print("========= Testing =========")
    #     model.load_model()
    #     model.test_corpus()
    # if args.mode == "test":
    #     model.test_corpus()
