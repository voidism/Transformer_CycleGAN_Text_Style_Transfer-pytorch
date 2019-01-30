from new_trans import *
from char_cnn_discriminator import *
import argparse
seq_len = 17

def continuous_decode(model, src, src_mask, max_len, start_symbol=2):
    memory = model.encode(src, src_mask) # encode is discrete
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data).to(device)
    collect_out = model.tgt_embed[0](ys).float()
    # word_col = ys[:]
    # ys = model.tgt_embed(ys)
    for i in range(max_len-1):
        out = model.conti_decode(memory, src_mask, Variable(model.tgt_embed[1](collect_out)), 
                           Variable(subsequent_mask(collect_out.size(1))
                                    .type_as(src.data)))
        # collect_out.append(out[:, -1])
        collect_out = torch.cat([collect_out, out[:, -1].unsqueeze(1)], dim=1)
        # prob = model.generator(out[:, -1])
        # _, next_word = torch.max(prob, dim = 1)
        # word_col = torch.cat([word_col, next_word.unsqueeze(-1)], dim=1)
        # ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        # ys = torch.cat([ys, model.src_embed(next_word.unsqueeze(1))], dim=1)
        # ys = torch.cat([ys, out[:, -1].unsqueeze(1)], dim=1)
    return collect_out#ys  #, word_col
    
def decode_with_output(model, src, src_mask, max_len, start_symbol=2):
    memory = model.encode(src, src_mask) # encode is discrete
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data).to(device)
    collect_out = model.tgt_embed(ys)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        collect_out = torch.cat([collect_out, out[:, -1].unsqueeze(1)], dim=1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
    return ys, collect_out

def prob_backward(model, src, src_mask, max_len, start_symbol=2):
    memory = model.encode(src, src_mask)
    ys = torch.ones(src.shape[0], 1).fill_(start_symbol).type_as(src.data).to(device)
    ret_back = model.tgt_embed[0].pure_emb(ys).float()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                        Variable(ys), 
                        Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1], scale=10)
        back = torch.matmul(prob ,model.tgt_embed[0].lut.weight.data.float())
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        ret_back = torch.cat([ret_back, back.unsqueeze(1)], dim=1)
    return ret_back

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

class CycleGAN(nn.Module):
    def __init__(self, discriminator, generator, utils):
        super(CycleGAN, self).__init__()
        self.D = discriminator
        self.G = generator
        self.R = copy.deepcopy(generator)
        self.D_opt = torch.optim.Adam(self.D.parameters())
        self.G_opt = torch.optim.Adam(self.G.parameters())
        self.R_opt = torch.optim.Adam(self.R.parameters())

        self.utils = utils
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse = nn.MSELoss()

    def save_model(self, d_path="Dis_model.ckpt", g_path="Gen_model.ckpt", r_path="Res_model.ckpt"):
        torch.save(self.D.state_dict(), d_path)
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.R.state_dict(), r_path)

    def load_model(self, path="", g_file=None, d_file=None, r_file=None):
        if g_file!=None:
            self.D.load_state_dict(torch.load(os.path.join(path, g_file)))
        if d_file!=None:
            self.G.load_state_dict(torch.load(os.path.join(path, d_file)))
        if r_file!=None:
            self.R.load_state_dict(torch.load(os.path.join(path, r_file)))
        print("model loaded!")


    def train_model(self, num_epochs=100, d_steps=50, g_steps=70, main_device="cuda:0", sec_device="cuda:1"):
        # self.D.to(self.D.device)
        # self.G.to(self.G.device)
        # self.R.to(self.R.device)
        X_datagen = self.utils.data_generator("X")
        Y_datagen = self.utils.data_generator("Y")
        for epoch in range(num_epochs):
            d_ct = Clock(d_steps, title="Train Discriminator(%d/%d)"%(epoch, num_epochs))
            for i, X_data, Y_data in zip(range(d_steps), data_gen(X_datagen, self.utils.sents2idx), data_gen(Y_datagen, self.utils.sents2idx)):
                # 1. Train D on real+fake
                self.D.zero_grad()
        
                #  1A: Train D on real
                d_real_pred = self.D(self.G.tgt_embed[0](Y_data.src))
                d_real_error = self.criterion(d_real_pred, torch.ones((d_real_pred.shape[0],), dtype=torch.int64).to(self.D.device))  # ones = true
                d_real_error.backward() # compute/store gradients, but don't change params
                self.D_opt.step()

                #  1B: Train D on fake
                self.G.to(main_device)
                d_fake_data = prob_backward(self.G, X_data.src, X_data.src_mask, max_len=seq_len).detach()  # detach to avoid training G on these labels
                d_fake_pred = self.D(d_fake_data)
                d_fake_error = self.criterion(d_fake_pred, torch.zeros((d_fake_pred.shape[0],), dtype=torch.int64).to(self.D.device))  # zeros = fake
                d_fake_error.backward()
                self.D_opt.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
                d_ct.flush(info={"D_loss":d_fake_error.item()})

            g_ct = Clock(g_steps, title="Train Generator(%d/%d)"%(epoch, num_epochs))
            for i, X_data in zip(range(g_steps), data_gen(X_datagen, self.utils.sents2idx)):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                self.G.zero_grad()
                g_fake_data = prob_backward(self.G, X_data.src, X_data.src_mask, max_len=seq_len)
                dg_fake_pred = self.D(g_fake_data)
                g_error = self.criterion(dg_fake_pred, torch.ones((dg_fake_pred.shape[0],), dtype=torch.int64).to(self.D.device))  # we want to fool, so pretend it's all genuine
        
                g_error.backward(retain_graph=True)
                self.G_opt.step()  # Only optimizes G's parameters
                self.G.zero_grad()
                g_ct.flush(info={"G_loss": g_error.item()})

                # 3. reconstructor
                r_reco_data = reconstruct(self.R, g_fake_data, max_len=seq_len)
                x_orgi_data = self.R.tgt_embed[0].pure_emb(X_data.src)
                r_error = self.mse(r_reco_data.float(), x_orgi_data.float())
                r_error.backward()
                self.R.zero_grad()
                self.R_opt.step()
                self.G_opt.step()
                g_ct.flush(info={"G_loss": g_error.item(),
                "R_loss": r_error.item()})
                
            with torch.no_grad():
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
            self.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="execute mode")
    # parser.add_argument("-filename", default=None, required=False, help="test filename")
    # parser.add_argument("-actual_name", default=None, required=False, help="test filename")
    # parser.add_argument("-load_model", default=False, required=False, help="test filename")
    # parser.add_argument("-model_name", default="model.ckpt", required=False, help="test filename")
    # parser.add_argument("-save_path", default="", required=False, help="test filename")
    # parser.add_argument("-train_file", default=None, required=False, help="test filename")
    # parser.add_argument("-test_file", default=None, required=True, help="test filename")
    # parser.add_argument("-epoch", default=1, required=False, help="test filename")
    args = parser.parse_args()

    utils = Utils(X_data_path="small_cou.txt", Y_data_path="small_cna.txt")
    # Train the simple copy task.
    V = 10000
    # _,_, emb_mat = load_embedding(limit=100000)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, utils.emb_mat, utils.emb_mat)
    d = Discriminator(utils.emb_mat.shape[1], int(utils.emb_mat.shape[1]/2), 15+2)
    cyclegan = CycleGAN(d, model, utils)
    cyclegan.D.src_embed = cyclegan.D.tgt_embed = cyclegan.R.src_embed = cyclegan.R.tgt_embed
    # cyclegan.D = torch.nn.DataParallel(cyclegan.D, device_ids=[0, 1]).cuda().module
    cyclegan.G.load_model(filename="model_9.ckpt")
    # cyclegan.G = torch.nn.DataParallel(cyclegan.G, device_ids=[0, 1]).cuda().module
    cyclegan.R.load_model(filename="model_9.ckpt")
    cyclegan = torch.nn.DataParallel(cyclegan, device_ids=[0, 1]).cuda().module
    if args.mode == "train":
        cyclegan.train_model()