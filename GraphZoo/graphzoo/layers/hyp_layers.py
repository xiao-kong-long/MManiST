"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from GraphZoo.graphzoo.layers.att_layers import DenseAtt

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer
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
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer
    """

    def __init__(self, manifold, in_features, out_features, r, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        # self.c = c
        self.radius = r
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):

        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        # init.xavier_normal_(self.weight, gain=1.414)
        init.constant_(self.bias, 0)

    def curvature(self):
        return 1. / self.radius.pow(2)

    def forward(self, x):
        self.c = self.curvature()
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        self.c = self.curvature()
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer
    """

    def __init__(self, manifold, r, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        # self.c = c
        self.radius = r

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
        
        self.hidden_dim = int(in_features/2)
        self.weight = nn.Parameter(torch.Tensor(in_features,self.hidden_dim))
        self.alpha = nn.Parameter(torch.Tensor(2*self.hidden_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.xavier_uniform_(self.alpha, gain=math.sqrt(2))
        

    def curvature(self):
        return 1. / self.radius.pow(2)

    def forward(self, x, adj):
        self.c = self.curvature()
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                # adj_att = self.att(x_tangent, adj)
                
                # attention by dot product
                # full_att = torch.matmul(x_tangent, x_tangent.T)
                # norm = torch.unsqueeze(torch.norm(x_tangent, 2, 1), 1)
                # norm_mat = torch.matmul(norm, norm.T)
                # full_att = full_att / norm_mat
                # zeros = torch.zeros_like(full_att)
                # adj_dist = torch.where(adj>0, full_att, zeros)
                # # normalize
                # diag = torch.diag(adj_dist)
                # adj_dist = adj_dist - torch.diag_embed(diag)
                # rowsum = torch.sum(adj_dist, dim=1)
                # adj_att = adj_dist / rowsum
                # adj_att = adj_att + torch.eye(n=adj.shape[0]).to('cuda:0')
                
                # attention by euclidean distance
                edge = adj.nonzero()
                eu_dists = 1/(torch.norm((x_tangent[edge[:,0],:] - x_tangent[edge[:,1], :]), 2, 1) + 1E5)
                adj_dist = torch.zeros_like(adj)
                adj_dist.index_put_((edge[:, 0], edge[:, 1]), eu_dists)
                diag = torch.diag(adj_dist)
                adj_dist = adj_dist - torch.diag_embed(diag)
                rowsum = torch.sum(adj_dist, dim=1)
                adj_att = adj_dist / rowsum
                adj_att = adj_att + torch.eye(n=adj.shape[0]).to('cuda:0')
                
                # attention by GAT model
                # edge = adj.nonzero()
                # x_src = x_tangent[edge[:,0],:]
                # x_dst = x_tangent[edge[:,1],:]
                # x_src = torch.matmul(x_src, self.weight)
                # x_dst = torch.matmul(x_dst, self.weight)
                # x = torch.cat((x_src, x_dst), 1)
                # att_list = torch.matmul(x, self.alpha).squeeze()
                
                # adj_att = torch.zeros_like(adj)
                # adj_att.index_put_((edge[:, 0], edge[:, 1]), att_list)



                adj_att = 0.5*adj_att + 0.5*adj
                support_t = torch.matmul(adj_att, x_tangent)

        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        self.c = self.curvature()
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer
    """

    def __init__(self, manifold, r_in, r_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        # self.c_in = c_in
        # self.c_out = c_out
        self.raduis_in = r_in
        self.raduis_out = r_out
        self.act = act

    def curvature_in(self):
        return 1. / self.raduis_in.pow(2)
    
    def curvature_out(self):
        return 1. / self.raduis_out.pow(2)

    def forward(self, x):
        self.c_in = self.curvature_in()
        self.c_out = self.curvature_out()
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        self.c_in = self.curvature_in()
        self.c_out = self.curvature_out()
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
