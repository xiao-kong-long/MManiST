import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data, gain=math.sqrt(2))
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    

class DGI_loss(torch.nn.Module):
    def __init__(self, in_features):  #
        super(DGI_loss, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))  # hidden_channels
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        self.EPS = 1e-8

        self.disc = Discriminator(in_features)

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        # print(torch.sigmoid(value))
        return torch.sigmoid(value) if sigmoid else value
        
    def forward(self, h_pos, h_neg, adj):
        
        summary = torch.sigmoid(h_pos.mean(dim=0))
        
        # global_emb = torch.matmul(adj, h_pos)
        # global_emb = F.normalize(global_emb, p=2, dim=1)
        # global_emb = torch.sigmoid(global_emb)

        # global_emb_a = torch.matmul(adj, h_neg)
        # global_emb_a = F.normalize(global_emb_a, p=2, dim=1)
        # global_emb_a = torch.sigmoid(global_emb_a)

        # ret = self.disc(global_emb, h_pos, h_neg)
        # ret_a = self.disc(global_emb_a, h_neg, h_pos)
        
        # self.loss_CSL = nn.BCEWithLogitsLoss()
        # n_spot = h_pos.shape[0]
        # one_matrix = np.ones([n_spot, 1])
        # zero_matrix = np.zeros([n_spot, 1])
        # label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
        # label_CSL = torch.FloatTensor(label_CSL).to('cuda:0')

        # sl_loss_1 = self.loss_CSL(ret, label_CSL)
        # sl_loss_2 = self.loss_CSL(ret_a, label_CSL)
        
        pos_loss = -torch.log(self.discriminate(h_pos, summary) + self.EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(h_neg, summary) + self.EPS).mean()
        
        return pos_loss + neg_loss
        # return sl_loss_1 + sl_loss_2