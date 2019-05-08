import re
import torch
from torch import nn
import os
from ELMoForManyLangs import elmo
import numpy as np

class Embedder(nn.Module):
    def __init__(self, seq_len=0, use_cuda=True, run_device=None, target_device=None ,d_model=1024):
        super(Embedder, self).__init__()
        self.embedder = elmo.Embedder(model_dir="new.model", batch_size=512, use_cuda=use_cuda)
        self.seq_len = seq_len
        self.device = run_device
        self.target_device = target_device
        if self.device != None:
            self.embedder.model.to(self.device)
        self.bos_vec, self.eos_vec, self.pad, self.oov = self.embedder.sents2elmo([["<bos>","<eos>","<pad>","<oov>"]], output_layer=0)[0]

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

    def forward(self, sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False):
        return torch.from_numpy(self.__call__(sents, max_len=0, with_bos_eos=True, layer=-1, pad_matters=False)[0]).to(self.target_device)

    def sub_unk(self, e):
        e = e.replace('，',',')
        e = e.replace('：',':')
        e = e.replace('；',';')
        e = e.replace('？','?')
        e = e.replace('！', '!')
        return e

class oov_handler():
    def __init__(self):
        self.ch_gex = re.compile(r'[\u4e00-\u9fff]+')
        self.num_gex = re.compile(r'[0-9]+')
        self.eng_gex = re.compile(r'[a-zA-Z]+')
        self.sym_list = list(np.load(os.path.join(os.path.dirname(__file__),"sym_list.npy")))
    def __call__(self, word):
        if self.ch_gex.findall(word) != []:
            return "<oov>"
        if self.eng_gex.findall(word) != []:
            return "<eng>"
        if self.num_gex.findall(word) != []:
            return "<num>"
        if word in self.sym_list:
            return "<sym>"
        else:
            return "<unk>"

class invELMo(nn.Module):
    def __init__(self,
            elmo=None,
            batch_size=32,
            input_size=1024,
            hidden_size=300,
            h_size=500,
            n_layers=3,
            dropout=0.33):
        super(invELMo, self).__init__()
        self.batch_size = batch_size
        self.vocab_lim = 100000
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # self.load_elmo()
        if elmo == None:
            print("ELMo model not provided. You can't use this model to train, but you can test.")
        self.elmo = elmo
        self.total_line = 50563844
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,
                          batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, self.vocab_lim)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_corpus_dict(self.vocab_lim)
        self.handle_oov = oov_handler()
        self.bos_vec, self.eos_vec = np.load(os.path.join(os.path.dirname(__file__),"bos_eos.npy"))

    def process(self, sent):
        return [x if x in self.word2idx else self.handle_oov(x) for x in sent]

    def load_elmo(self):
        print("loading ELMo model ...")
        self.elmo = Embedder()
        print("ELMo model loaded!")

    def load_corpus_dict(self, limit):
        self.idx2word = ["<pad>"] + list(np.load(os.path.join(os.path.dirname(__file__),"idx2word_new.npy")))[:limit-1]
        self.word2idx = dict([(word, i) for i, word in enumerate(self.idx2word)])

    def corpus_generator(self, filename="shuff_corpus.txt"):
        f = open(os.path.join(os.path.dirname(__file__),filename))
        batch_list = []
        for i in f:
            batch_list.append(self.process(["<bos>"]+sub_unk(i.strip()).split(' ')+["<eos>"]))
            if len(batch_list) == self.batch_size:
                yield batch_list
                batch_list = []
            

    def padded_corpus_generator(self, filename="shuff_corpus.txt", max_len=25):
        f = open(os.path.join(os.path.dirname(__file__),filename))
        batch_list = []
        for i in f:
            org_sent = sub_unk(i.strip()).split(' ')
            if len(org_sent) > max_len - 2:
                org_sent = org_sent[:max_len - 2]
            batch_list.append(self.process(["<bos>"] + org_sent + ["<eos>"] + ["<pad>" for _ in range(max_len - 2 - len(org_sent))]))
            if len(batch_list) == self.batch_size:
                yield batch_list
                batch_list = []


    def sent2idx(self, sents, max_len = 0):
        if max_len==0:
            for i in sents:
                if len(i) > max_len:
                    max_len = len(i)
        sents_lens = []
        sent_mat = np.zeros((len(sents), max_len), dtype=np.int64)
        for i in range(len(sents)):
            sents_i_len = len(sents[i])
            sents_lens.append(sents_i_len)
            for j in range(max_len):
                if j < sents_i_len:
                    sent_mat[i][j] = self.word2idx[sents[i][j]]
        return [sent_mat, np.array(sents_lens)]

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = torch.from_numpy(input_seq).to(self.device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        pred_prob = self.fc1(outputs)#nn.Softmax(dim=-1)(self.fc1(outputs))
        embedded.cpu()
        return pred_prob

    def train_model(self, num_epochs=1, step_num=100000, step_to_save_model=1000, filename="shuff_corpus.txt"):
        self.to(self.device)
        for epoch in range(num_epochs):
            ct = Clock(step_num)
            His_loss = History(title="Loss", xlabel="step", ylabel="loss",
            item_name=["train_loss"])
            His_ppl = History(title="Perplexity", xlabel="step", ylabel="loss",
            item_name=["train_ppl"])
            for step, batch_x in enumerate(self.padded_corpus_generator(filename=filename)):
                batch_y, x_lens = self.sent2idx(batch_x)
                elmo_x, x_lens = self.elmo(batch_x)
                (elmo_x, x_lens, batch_y), _ind = sort_numpy([elmo_x, x_lens, batch_y], piv=1)
                target = torch.from_numpy(batch_y).cuda() if self.device!="cpu" else torch.from_numpy(batch_y)
                pred = self.forward(elmo_x, x_lens)
                loss = self.criterion(pred.transpose(1, 2), target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ppl = math.exp(loss.cpu().item())
                info_dict = {"loss": loss, "ppl": ppl}
                ct.flush(info=info_dict)
                His_loss.append_history(0, (step, loss))
                His_ppl.append_history(0, (step, ppl))
                target.cpu()
                if step == step_num:
                    break
                if (step + 1) % step_to_save_model == 0:
                    torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__),'model.ckpt'))
                    His_loss.plot("loss_plot")
                    His_ppl.plot("pll_plot")
                    test_corpus(self, "small_cou.txt")
            torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__),'model.ckpt'))
            His_loss.plot()
            His_ppl.plot()


    def load_model(self, filename='model.ckpt', device="cuda:0"):
        self.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),filename), map_location=device))
        print("model.ckpt load!")

    def test(self, input_seq, input_lengths, hidden=None):
        pred_prob = self.forward(input_seq, input_lengths)
        pred_idx = pred_prob.argmax(2)
        return [[self.idx2word[x] for x in r] for r in pred_idx]