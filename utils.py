from attn import *

def load_embedding(limit=100000):
    idx2word = ["<pad>","<unk>"] + list(np.load(os.path.join(os.path.dirname(__file__),"WordEmb/idx2word.npy")))[:limit-2]
    word2idx = dict([(word, i) for i, word in enumerate(idx2word)])
    syn0 = np.load("WordEmb/word2vec_weights.npy")[:limit - 2]
    syn0 = np.concatenate((np.zeros((2, syn0.shape[1])), syn0),axis=0)
    chgex = re.compile(r'[\u4e00-\u9fff]+')
    non_ch = []
    for i in range(2, len(idx2word)):
        if chgex.findall(idx2word[i]).__len__()==0:
                non_ch.append(syn0[i])
    syn0[1] = np.mean(non_ch, axis=0)
    # syn0[1] = non_ch[87]
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

class Utils():
    def __init__(self,
    X_data_path,
    Y_data_path,
    X_test_path,
    Y_test_path,
    batch_size = 32, vocab_lim=10000):
        self.X_data_path = X_data_path
        self.X_line_num = int(os.popen("wc -l %s"%self.X_data_path).read().split(' ')[0])
        self.Y_data_path = Y_data_path
        self.Y_line_num = int(os.popen("wc -l %s" % self.Y_data_path).read().split(' ')[0])
        
        self.X_test_path = X_test_path
        self.X_test_num = int(os.popen("wc -l %s"%self.X_test_path).read().split(' ')[0])
        self.Y_test_path = Y_test_path
        self.Y_test_num = int(os.popen("wc -l %s" % self.Y_test_path).read().split(' ')[0])
        
        self.idx2word, self.word2idx, self.emb_mat = load_embedding(limit=vocab_lim)
        self.batch_size = batch_size
        self.train_step_num = math.floor(self.X_line_num / batch_size)
        self.test_step_num = math.floor(self.X_test_num / batch_size)
        self.device = "cuda:0"
        self.ch_gex = re.compile(r'[\u4e00-\u9fff]+')
        self.eng_gex = re.compile(r'[a-zA-Z0-9０１２３４５６７８９\s]+')
        self.max_len = 15
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
        # char_list = self.string2list("".join(word_list))
        # for i, char in enumerate(char_list):
        #     if char not in self.word2idx:
        #         char_list[i] = "<unk>"
        for i, word in enumerate(word_list):
            if word not in self.word2idx:
                word_list[i] = "<unk>"
        return word_list

    def data_generator(self, mode="X", write_actual_data=False):
        path = eval("self.%s_data_path" % mode)
        while True:
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

    def test_generator(self, mode="X", write_actual_data=False):
        path = eval("self.%s_test_path" % mode)
        while True:
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