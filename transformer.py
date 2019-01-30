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

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # Embedding function
        self.tgt_embed = tgt_embed # Embedding function
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def load_model(self, filename='model.ckpt', device="cuda:0"):
        self.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__),filename), map_location=device))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # layer = EncoderLayer()
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask.to(device), 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pre_trained_matrix):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.lut.weight.data = torch.tensor(pre_trained_matrix)
        self.lut.weight.requires_grad = False
        self.lut.to(device)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.to(device)) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x.float() + self.pe[:, :x.size(1)]
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, src_pre_trained_mat, tgt_pre_trained_mat, N=6, 
               d_model=1024, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab, src_pre_trained_mat), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab, tgt_pre_trained_mat), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def load_embedding(limit=100000):
    idx2word = ["<pad>","<unk>"] + list(np.load(os.path.join(os.path.dirname(__file__),"CharEmb/idx2word.npy")))[:limit-2]
    word2idx = dict([(word, i) for i, word in enumerate(idx2word)])
    syn0 = np.load("CharEmb/word2vec_weights.npy")[:limit-2]
    syn0 = np.concatenate((np.zeros((2, syn0.shape[1])), syn0),axis=0)
    return idx2word, word2idx, syn0

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

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

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

def data_gen(iterator, sent2idx):
    "Generate random data for a src-tgt copy task."
    for i in iterator:
        data = torch.from_numpy(sent2idx(i)).long()
        # data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # data[:, 0] = 1
        data = torch.cat((torch.full((data.shape[0], 1), 2, dtype=torch.long), data), dim=1)
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)).to(device), 
                              y.contiguous().view(-1)).to(device) / norm.float().to(device)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # print("ddd:", loss.data)
        return loss.data.to(device) * norm.float().to(device)

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def greedy_decode(model, src, src_mask, max_len, start_symbol=2):
    memory = model.encode(src, src_mask)
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data).to(device)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    return ys

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).to(device), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze().to(device), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Utils():
    def __init__(self,
    X_data_path,
    Y_data_path,
    batch_size = 32, vocab_lim=10000):
        self.X_data_path = X_data_path
        self.X_line_num = int(os.popen("wc -l %s"%self.X_data_path).read().split(' ')[0])
        self.Y_data_path = Y_data_path
        self.Y_line_num = int(os.popen("wc -l %s"%self.Y_data_path).read().split(' ')[0])
        self.idx2word, self.word2idx, self.emb_mat = load_embedding(limit=vocab_lim)
        self.batch_size = batch_size
        self.train_step_num = math.floor(self.X_line_num / batch_size)
        self.test_step_num = math.floor(self.Y_line_num / batch_size)
        self.device = "cuda:0"
        self.ch_gex = re.compile(r'[\u4e00-\u9fff]+')
        self.eng_gex = re.compile(r'[a-zA-Z0-9０１２３４５６７８９\s]+')
        self.max_len = 40
        self.vocab_lim = vocab_lim

    def string2list(self, line):
        ret = []
        temp_str = []
        for char in line:
            if self.eng_gex.findall(char).__len__() == 0:
                if temp_str.__len__() > 0:
                    ret.append("".join(temp_str).strip())
                    temp_str = []
                ret.append(char)
            else:
                temp_str.append(char)
        if temp_str.__len__() > 0:
            ret.append("".join(temp_str).strip())
        return ret

    def process_sent(self, sent):
        sent = f2h(sent)
        word_list = re.split(r"[\s|\u3000]+", sent.strip())
        char_list = self.string2list("".join(word_list))
        for i, char in enumerate(char_list):
            if char not in self.word2idx:
                char_list[i] = "<unk>"
        return char_list

    def data_generator(self, mode="X", write_actual_data=False):
        if write_actual_data:
            fw = open("actual_test_data.utf8", 'w')
        path = eval("self.%s_data_path" % mode)
        file = open(path)
        sents = []
        for sent in file:
            if len(sent.strip()) == 0:
                continue
            word_list = self.process_sent(sent)
            sents.append(word_list)
            if len(sents) == self.batch_size:
                yield sents
                sents = []
        if len(sents)!=0:
            yield sents

    def sents2idx(self, sents, pad=0, add_eos=True, eos=3):
        idx_mat = np.zeros((len(sents), self.max_len + 1), dtype=np.int32) + pad
        for i in range(len(sents)):
            for j in range(min(len(sents[i]), self.max_len)):
                idx_mat[i][j] = self.word2idx[sents[i][j]]
            eos_pos = min(len(sents[i]), self.max_len)
            idx_mat[i][eos_pos] = eos
        return idx_mat

    def idx2sent(self, idxs, pad=0):
        ret = []
        for i in range(len(idxs)):
            sent = []
            for j in range(len(idxs[i])):
                sent.append(self.idx2word[idxs[i][j]])
            ret.append(sent)
        return ret

if __name__ == "__main__":
    utils = Utils(X_data_path="small_cou.txt", Y_data_path="small_cna.txt")
    # Train the simple copy task.
    V = utils.emb_mat.shape[0]
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, utils.emb_mat, utils.emb_mat)
    model.to("cuda:0")
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    X_test_batch = None
    Y_test_batch = None
    for i, batch in enumerate(data_gen(utils.data_generator("X"), utils.sents2idx)):
        X_test_batch = batch
        break

    for i, batch in enumerate(data_gen(utils.data_generator("Y"), utils.sents2idx)):
        Y_test_batch = batch
        break

    if sys.argv[1] == "train":
        for epoch in range(int(sys.argv[2])):
            model.train()
            print("EPOCH %d:"%(epoch+1))
            pretrain_run_epoch(data_gen(utils.data_generator("Y"), utils.sents2idx), model, 
                    SimpleLossCompute(model.generator, criterion, model_opt), utils.train_step_num)
            model.eval()

            x = utils.idx2sent(greedy_decode(model, X_test_batch.src, X_test_batch.src_mask, max_len=20, start_symbol=2))
            y = utils.idx2sent(greedy_decode(model, Y_test_batch.src, Y_test_batch.src_mask, max_len=20, start_symbol=2))

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

            # print(pretrain_run_epoch(data_gen(utils.data_generator("X"), utils.sents2idx), model, 
            #                 SimpleLossCompute(model.generator, criterion, None), utils.train_step_num))
    
        torch.save(model.state_dict(), 'model.ckpt')