import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import numpy as np


class GraphConvolution2(Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, bias=False):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))

    def forward(self, input):
        x, adj = input
        z = F.dropout(x, self.dropout, self.training)
        z = torch.mm(z, self.weight)
        z = torch.mm(adj, z)
        
        return z, adj


class AttentionLayer(Module):
    def __init__(self, in_features, dropout=0.0, act=F.relu, bias=False):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.project = nn.Linear(in_features, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta*z).sum(1)

class SelfAttention(Module):
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.matmul(q, k.transpose(1,2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values.sum(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x.sum(1)
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, b, m = x.size()
        y = self.avg_pool(x).view(n, b)
        y = self.fc(y).view(n, b, 1)
        return (x * y).sum(1)


class CoGuidedLayer(Module):
    def __init__(self, in_features, dropout=0.0, act=F.relu, bias=False):
        super(CoGuidedLayer, self).__init__()

        self.in_features = in_features

        self.w_pi_1 = nn.Linear(in_features, in_features, bias=True)
        self.w_pi_2 = nn.Linear(in_features, in_features, bias=True)

        self.w_eu_z = nn.Linear(in_features, in_features, bias=True)
        self.w_eu_r = nn.Linear(in_features, in_features, bias=True)
        self.w_eu = nn.Linear(in_features, in_features, bias=True)
        self.u_eu = nn.Linear(in_features, in_features, bias=True)
        self.v_eu = nn.Linear(in_features, in_features, bias=True)

        self.w_po_z = nn.Linear(in_features, in_features, bias=True)
        self.w_po_r = nn.Linear(in_features, in_features, bias=True)
        self.w_po = nn.Linear(in_features, in_features, bias=True)
        self.u_po = nn.Linear(in_features, in_features, bias=True)
        self.v_po = nn.Linear(in_features, in_features, bias=True)

        self.w_lo_z = nn.Linear(in_features, in_features, bias=True)
        self.w_lo_r = nn.Linear(in_features, in_features, bias=True)
        self.w_lo = nn.Linear(in_features, in_features, bias=True)
        self.u_lo = nn.Linear(in_features, in_features, bias=True)
        self.v_lo = nn.Linear(in_features, in_features, bias=True)

    def forward(self, h_eu, h_po, h_lo):
        m_c = torch.tanh(self.w_pi_1(h_eu * h_po * h_lo))
        m_j = torch.tanh(self.w_pi_2(h_eu + h_po + h_lo))

        r_eu = torch.sigmoid(self.w_eu_z(m_c) + self.w_eu_r(m_j))
        r_po = torch.sigmoid(self.w_po_z(m_c) + self.w_po_r(m_j))
        r_lo = torch.sigmoid(self.w_lo_z(m_c) + self.w_lo_r(m_j))

        m_eu = torch.tanh(self.w_eu(h_eu * r_eu) + self.u_eu((1 - r_eu) * h_po)) + self.v_eu((1 - r_eu) * h_lo)
        m_po = torch.tanh(self.w_po(h_po * r_po) + self.u_po((1 - r_po) * h_lo)) + self.v_po((1 - r_po) * h_eu)
        m_lo = torch.tanh(self.w_lo(h_lo * r_lo) + self.u_lo((1 - r_lo) * h_eu)) + self.v_lo((1 - r_lo) * h_po)

        h_eu_ = m_eu * (h_eu + m_po + m_lo)
        h_po_ = m_po * (h_po + m_lo + m_eu)
        h_lo_ = m_lo * (h_lo + m_eu + m_po)
        
        return h_eu_ + h_po_ + h_lo_
    

class InterViewAttention(Module):
    def __init__(self, in_features, dropout=0.0, act=F.relu, bias=False):
        super(InterViewAttention, self).__init__()

        self.a1 = Parameter(torch.FloatTensor(in_features, 1))
        self.a2 = Parameter(torch.FloatTensor(in_features, 1))
        self.a3 = Parameter(torch.FloatTensor(in_features, 1))
        self.gamma = 0.5
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.a1, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.a2, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.a3, gain=math.sqrt(2))

    def forward(self, h_eu, h_po, h_lo, adj):
        N = h_eu.size()[0]

        e_eu = torch.matmul(h_eu, self.a1)
        e_po = torch.matmul(h_po, self.a2).t()
        e_lo = torch.matmul(h_lo, self.a3)
        
        e = e_eu + e_po + e_lo
        e = F.leaky_relu(e)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = torch.mul(attention, adj.sum(1).repeat(N, 1).t())
        attention = torch.add(attention * self.gamma, adj * (1 - self.gamma))

        h_prime = torch.matmul(attention, h_po)

        return h_prime