"""Euclidean layers"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
import numpy as np

def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
    return dims, acts


class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        if adj.is_sparse:
            support = torch.spmm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        output = self.act(support), adj
        return output

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )

class GraphConvolution1(Module):
    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution1, self).__init__()
        self.dropout = dropout
        self.conv = GCNConv(in_features, out_features, add_self_loops=False, normalize=False)
        self.act = act

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        adj_sp = adj.to_sparse()
        
        if adj_sp.is_cuda:
            edge_index = torch.LongTensor(adj_sp.indices().cpu()).cuda()
        else:
            edge_index = torch.LongTensor(adj_sp.indices())
        x1 = self.act(self.conv(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout)

        return x1, adj

    def extra_repr(self):
        return 'input_dim={}, output_dim={}'.format(
                self.in_features, self.out_features
        )


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


class Linear2(Module):
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu, bias=False):
        super(Linear2, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.bias = bias
        
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias_weight = Parameter(torch.FloatTensor(1, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.bias_weight, gain=math.sqrt(2))

    def forward(self, x):

        z = F.dropout(x, self.dropout, self.training)
        z = torch.mm(z, self.weight)
        
        if self.bias:
            z = z + self.bias_weight

        return z


class Linear(Module):
    """
    Simple Linear layer with dropout
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances"""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

