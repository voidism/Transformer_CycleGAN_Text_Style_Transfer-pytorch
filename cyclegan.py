import re
import torch
from torch import nn
import os
from ELMoForManyLangs import elmo
import numpy as np
from jexus import Clock, History
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
import sys
import random, time
cwd = os.getcwd()
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(cwd, "../InverseELMo"))
sys.path.append(os.path.join(cwd, "../CycleGAN-sentiment-transfer"))
from invELMo import invELMo

def f2h(s):
    s = list(s)
    for i in range(len(s)):
        num = ord(s[i])
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        s[i] = chr(num).translate(str.maketrans('﹕﹐﹑。﹔﹖﹗﹘　', ':,、。;?!- '))
    return re.sub(r"( |　)+", " ", "".join(s)).strip()
    
def sort_list(li, piv=2,unsort_ind=None):
    ind = []
    if unsort_ind == None:
        ind = sorted(range(len(li[piv])), key=(lambda k: li[piv][k]))
    else:
        ind = unsort_ind
    for i in range(len(li)):
        li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_numpy(li, piv=2,unsort=False):
    ind = np.argsort(-li[piv] if not unsort else li[piv], axis=0)
    for i in range(len(li)):
        if type(li[i]).__module__ == np.__name__ or type(li[i]).__module__ == torch.__name__:
            li[i] = li[i][ind]
        else:
            li[i] = [li[i][j] for j in ind]
    return li, ind

def sort_torch(li, piv=2,unsort=False):
    li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    for i in range(len(li)):
        if i == piv:
            continue
        else:
            li[i] = li[i][ind]
    return li, ind

def sort_by(li, piv=2, unsort=False):
    if type(li[piv]).__module__ == np.__name__:
        return sort_numpy(li, piv, unsort)
    elif type(li[piv]).__module__ == torch.__name__:
        return sort_torch(li, piv, unsort)
    else:
        return sort_list(li, piv, unsort)


class Embedder():
    def __init__(self, seq_len=0, use_cuda=True, device=None):
        self.embedder = elmo.Embedder(batch_size=512, use_cuda=use_cuda)
        self.seq_len = seq_len
        self.bos_vec, self.eos_vec = np.load("bos_eos.npy")
        self.pad, self.oov = np.load("pad_oov.npy")
        self.device = device
        if self.device != None:
            self.embedder.model.to(self.device)

    def __call__(self, sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False):
        seq_lens = np.array([len(x) for x in sents], dtype=np.int64)
        sents = [[self.sub_unk(x) for x in sent] for sent in sents]
        if max_len != 0:
            pass
        elif self.seq_len != 0:
            max_len = self.seq_len
        else:
            max_len = seq_lens.max()
        emb_list = self.embedder.sents2elmo(sents, output_layer=layer)
        if not with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([emb_list[i], np.tile(self.pad,[max_len - seq_lens[i],1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([emb_list[i], np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))])
                else:
                    emb_list[i] = emb_list[i][:max_len]
        elif with_bos_eos:
            for i in range(len(emb_list)):
                if max_len - seq_lens[i] > 0:
                    if pad_matters:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.tile(self.pad, [max_len - seq_lens[i], 1])], axis=0)
                    else:
                        emb_list[i] = np.concatenate([
                            self.bos_vec[np.newaxis],
                            emb_list[i],
                            self.eos_vec[np.newaxis],
                            np.zeros((max_len - seq_lens[i], emb_list[i].shape[1]))], axis=0)
                else:
                    emb_list[i] = np.concatenate([self.bos_vec[np.newaxis], emb_list[i][:max_len],self.eos_vec[np.newaxis]], axis=0)
        embedded = np.array(emb_list, dtype=np.float32)
        seq_lens = seq_lens+2 if with_bos_eos else seq_lens
        return embedded, seq_lens

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e


class Utils():
    def __init__(self,
    training_data_path,
    testing_data_path,
    batch_size = 32, elmo_device=None):
        self.training_data_path = training_data_path
        self.training_line_num = int(os.popen("wc -l %s"%self.training_data_path).read().split(' ')[0])
        self.testing_data_path = testing_data_path
        self.testing_line_num = int(os.popen("wc -l %s"%self.testing_data_path).read().split(' ')[0])
        self.elmo = Embedder(device=elmo_device, use_cuda=elmo_device!="cpu")
        self.batch_size = batch_size
        self.train_step_num = math.floor(self.training_line_num / batch_size)
        self.test_step_num = math.floor(self.testing_line_num / batch_size)
        self.device="cuda:0"

    def process_sent(self, sent):
        sent = f2h(sent)
        word_list = re.split(r"[\s|\u3000]+", sent.strip())
        char_list = list("".join(word_list))
        label_list = []
        for word in word_list:
            label_list += [0] * (len(word) - 1) + [1]
        return char_list, label_list

    def sent2list(self, sent):
        sent = f2h(sent)
        word_list = re.split(r"[\s|\u3000]+", sent.strip())
        return word_list

    def data_generator(self, mode="train", write_actual_data=False):
        if write_actual_data:
            fw = open("actual_test_data.utf8", 'w')
        path = eval("self.%sing_data_path" % mode)
        file = open(path)
        sents = []
        for sent in file:
            if len(sent.strip()) == 0:
                continue
            word_list = self.sent2list(sent)
            if len(word_list) > 150:# process long sentences
                continue
            else:
                if write_actual_data:
                    fw.write(' '.join(word_list) + '\n')
                sents.append(word_list)
                if len(sents) == self.batch_size:
                    yield sents
                    sents = []
        fw.close()
        if len(sents)!=0:
            yield sents

    def raw2elmo(self, batch, with_bos_eos=True):
        embedded, seq_lens = self.elmo(batch, with_bos_eos=with_bos_eos)
        return embedded, seq_lens

    def elmo2mask(self, embedded, seq_lens, with_bos_eos=True, mask_rate=0.0):
        # embedded, seq_lens = self.elmo(batch)
        mask = np.full((embedded.shape[0], embedded.shape[1]), -1, dtype=np.int64)
        if mask_rate:
            (begin, end) = (1, -1) if with_bos_eos else(0, 0)
            rand_mat = np.random.rand(embedded.shape[0], embedded.shape[1])
            for row in range(embedded.shape[0]):
                for col in range(begin, seq_lens[row] + end):
                    if rand_mat[row, col] < mask_rate:
                        embedded[row, col] = torch.zeros(embedded[row, col].shape)
                        mask[row, col] = 0
                    else:
                        mask[row, col] = 1
        return embedded, seq_lens, mask

class Generator(nn.Module):
    def __init__(self,
        batch_size=32,
        device="cuda:0",
        hidden_size=300,
        input_size=1024,
        encode_size=30,
            n_layers=3,
            dropout=0.33):
        super(Generator, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, input_size)
        self.hidden_expander_1 = nn.Linear(encode_size, hidden_size)
        self.hidden_expander_2 = nn.Linear(encode_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input_seq, input_lengths, hidden=None, sort=True, unsort=False):
        embedded = torch.from_numpy(input_seq).to(self.device)
        if sort:
            [embedded, input_lengths], ind = sort_by([embedded, input_lengths], piv=1)
        hidden[0] = self.hidden_expander_1(hidden[0])
        hidden[1] = self.hidden_expander_2(hidden[1])
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        pred_seq = self.fc1(outputs)#nn.Softmax(dim=-1)(self.fc1(outputs))
        embedded.cpu()
        if unsort:
            [pred_seq, _], _ = sort_by([pred_seq, ind], piv=1, unsort=True)
        return pred_seq

class Discriminator(nn.Module):
    def __init__(self,
        batch_size=32,
        device="cuda:0",
        hidden_size=300,
        input_size=1024,
            n_layers=3,
            dropout=0.33):
        super(Discriminator, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, 2)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input_seq, input_lengths, hidden=None, numpy=True, sort=True, unsort=False):
        if numpy:
            embedded = torch.from_numpy(input_seq).to(self.device)
        else:
            embedded = input_seq.to(self.device)
        if sort:
            [embedded, input_lengths], ind = sort_by([embedded, input_lengths], piv=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        pred_prob = self.fc1(outputs)#nn.Softmax(dim=-1)(self.fc1(outputs))
        embedded.cpu()
        if unsort:
            [pred_prob, _], _ = sort_by([pred_prob, ind], piv=1, unsort=True)
        return pred_prob


class Encoder(nn.Module):
    def __init__(self,
        batch_size=32,
        device="cuda:0",
        hidden_size=300,
        encode_size=30,
        input_size=1024,
            n_layers=3,
            dropout=0.33):
        super(Encoder, self).__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_size, encode_size)
        self.fc2 = nn.Linear(hidden_size, encode_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input_seq, input_lengths, hidden=None, sort=True, unsort=False):
        embedded = torch.from_numpy(input_seq).to(self.device)
        if sort:
            [embedded, input_lengths], ind = sort_by([embedded, input_lengths], piv=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, (h_n, c_n) = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        code1 = self.fc1(h_n)  #nn.Softmax(dim=-1)(self.fc1(outputs))
        code2 = self.fc2(c_n)
        embedded.cpu()
        if unsort:
            [code1, code2, _], _ = sort_by([code1, code2, ind], piv=2, unsort=True)
        return [code1, code2]

def load_cpu_invelmo():
    elmo = invELMo()
    elmo.device = "cpu"
    elmo.load_model(device="cpu")
    elmo.eval()
    return elmo

class MaskGAN():
    def __init__(self, embedder, discriminator, generator, encoder, utils):
        self.embedder = embedder
        self.D = discriminator
        self.G = generator
        self.encoder = encoder
        self.utils = utils
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.invelmo = load_cpu_invelmo()
        self.mse = nn.MSELoss()

    def save_model(self, d_path="Dis_model.ckpt", g_path="Gen_model.ckpt"):
        torch.save(self.D.state_dict(), d_path)
        torch.save(self.G.state_dict(), g_path)

    def load_model(self, path=""):
        self.D.load_state_dict(torch.load(os.path.join(path,"Dis_model.ckpt")))
        self.G.load_state_dict(torch.load(os.path.join(path,"Gen_model.ckpt")))
        print("model loaded!")

    def pretrain(self, num_epochs=1):
        self.G.to(self.G.device)
        self.encoder.to(self.encoder.device)
        real_datagen = self.utils.data_generator("train")
        test_datagen = self.utils.data_generator("test")
        for epoch in range(num_epochs):
            ct = Clock(self.utils.train_step_num, title="Pretrain(%d/%d)" % (epoch, num_epochs))
            for real_data in real_datagen:
                # 2. Train G on D's response (but DO NOT train D on these labels)
                self.G.zero_grad()

                g_org_data, g_data_seqlen = self.utils.raw2elmo(real_data)
                
                gen_input = self.encoder(g_org_data, g_data_seqlen)
                g_fake_data = self.G(g_org_data, g_data_seqlen, hidden=gen_input)
                loss = self.mse(g_fake_data, torch.from_numpy(g_org_data).to(self.G.device))
        
                loss.backward()
                self.G.optimizer.step()  # Only optimizes G's parameters
                self.encoder.optimizer.step()
                ct.flush(info={"G_loss": loss.item()})
                
            with torch.no_grad():
                for _, real_data in zip(range(2), test_datagen):
                    g_org_data, g_data_seqlen = self.utils.raw2elmo(real_data)
                    [g_org_data, g_data_seqlen], _ind = sort_by([g_org_data, g_data_seqlen], piv=1)
                    g_mask_data, g_data_seqlen, g_mask_label = \
                    self.utils.elmo2mask(g_org_data, g_data_seqlen, mask_rate=epoch/num_epochs)
                    gen_input = self.encoder(g_org_data, g_data_seqlen, sort=False)
                    g_fake_data = self.G(g_mask_data, g_data_seqlen, hidden=gen_input, sort=False)

                    gen_sents = self.invelmo.test(g_fake_data.cpu().numpy(), g_data_seqlen)
                    for i, j in zip(real_data, gen_sents):
                        print("="*50)
                        print(' '.join(i))
                        print("---")
                        print(' '.join(j))
                        print("=" * 50)
            torch.save(self.G.state_dict(), "pretrain_model.ckpt")


    def train_model(self, num_epochs=100, d_steps=10, g_steps=10):
        self.D.to(self.D.device)
        self.G.to(self.G.device)
        self.encoder.to(self.encoder.device)
        real_datagen = self.utils.data_generator("train")
        test_datagen = self.utils.data_generator("test")
        for epoch in range(num_epochs):
            d_ct = Clock(d_steps, title="Train Discriminator(%d/%d)"%(epoch, num_epochs))
            for d_step, real_data in zip(range(d_steps), real_datagen):
                # 1. Train D on real+fake
                self.D.zero_grad()
        
                #  1A: Train D on real
                d_org_data, d_data_seqlen = self.utils.raw2elmo(real_data)
                d_mask_data, d_data_seqlen, d_mask_label = \
                self.utils.elmo2mask(d_org_data, d_data_seqlen, mask_rate=epoch/num_epochs)
                d_real_pred = self.D(d_org_data, d_data_seqlen)
                d_real_error = self.criterion(d_real_pred.transpose(1, 2), torch.ones(d_mask_label.shape, dtype=torch.int64).to(self.D.device))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params
                self.D.optimizer.step()

                #  1B: Train D on fake
                d_gen_input = self.encoder(d_org_data, d_data_seqlen)
                d_fake_data = self.G(d_mask_data, d_data_seqlen, hidden=d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_pred = self.D(d_fake_data, d_data_seqlen, numpy=False)
                d_fake_error = self.criterion(d_fake_pred.transpose(1, 2), torch.from_numpy(d_mask_label).to(self.D.device))  # zeros = fake
                d_fake_error.backward()
                self.D.optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
                d_ct.flush(info={"D_loss":d_fake_error.item()})

            g_ct = Clock(g_steps, title="Train Generator(%d/%d)"%(epoch, num_epochs))
            for g_step, real_data in zip(range(g_steps), real_datagen):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                self.G.zero_grad()

                g_org_data, g_data_seqlen = self.utils.raw2elmo(real_data)
                g_mask_data, g_data_seqlen, g_mask_label = \
                self.utils.elmo2mask(g_org_data, g_data_seqlen, mask_rate=epoch/num_epochs)
                gen_input = self.encoder(g_org_data, g_data_seqlen)
                g_fake_data = self.G(g_mask_data, g_data_seqlen, hidden=gen_input)
                dg_fake_pred = self.D(g_fake_data, g_data_seqlen, numpy=False)
                g_error = self.criterion(dg_fake_pred.transpose(1, 2), torch.ones(g_mask_label.shape, dtype=torch.int64).to(self.D.device))  # we want to fool, so pretend it's all genuine
        
                g_error.backward()
                self.G.optimizer.step()  # Only optimizes G's parameters
                self.encoder.optimizer.step()
                g_ct.flush(info={"G_loss": g_error.item()})
                
            with torch.no_grad():
                for _, real_data in zip(range(2), test_datagen):
                    g_org_data, g_data_seqlen = self.utils.raw2elmo(real_data)
                    [g_org_data, g_data_seqlen], _ind = sort_by([g_org_data, g_data_seqlen], piv=1)
                    g_mask_data, g_data_seqlen, g_mask_label = \
                    self.utils.elmo2mask(g_org_data, g_data_seqlen, mask_rate=epoch/num_epochs)
                    gen_input = self.encoder(g_org_data, g_data_seqlen, sort=False)
                    g_fake_data = self.G(g_mask_data, g_data_seqlen, hidden=gen_input, sort=False)

                    gen_sents = self.invelmo.test(g_fake_data.cpu().numpy(), g_data_seqlen)
                    for i, j in zip(real_data, gen_sents):
                        print("="*50)
                        print(' '.join(i))
                        print("---")
                        print(' '.join(j))
                        print("=" * 50)
            self.save_model()
                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="execute mode")
    # parser.add_argument("-train_file", default=None, required=False, help="test filename")
    # parser.add_argument("-test_file", default=None, required=True, help="test filename")
    parser.add_argument("-load_model_path", default=None, required=False, help="test filename")
    parser.add_argument("-epoch", default=1000, required=False, help="test filename")
    parser.add_argument("-d_step", default=100, required=False, help="test filename")
    parser.add_argument("-g_step", default=100, required=False, help="test filename")
    args = parser.parse_args()


    embedder, discriminator, generator, encoder, utils = \
    Embedder(), Discriminator(), Generator(), Encoder(), \
    Utils(training_data_path="data/train_as.txt",
     testing_data_path="data/test_as.txt", elmo_device="cuda:0")
    model = MaskGAN(embedder, discriminator, generator, encoder, utils)
    if args.load_model_path != None:
        model.load_model(args.load_model_path)
    if args.mode == "train":
        model.train_model(num_epochs=int(args.epoch), d_steps=int(args.d_step), g_steps=int(args.g_step))
    if args.mode == "pretrain":
        model.pretrain(num_epochs=int(args.epoch))

