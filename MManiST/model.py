import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from GraphZoo.graphzoo.layers.layers import FermiDiracDecoder, Linear
from GraphZoo.graphzoo import manifolds
import GraphZoo.graphzoo.models.encoders as encoders
from GraphZoo.graphzoo.models.decoders import model2decoder

from GraphZoo.graphzoo.layers.hyp_att_layers import GraphAttentionLayer as HGATlayer
from GraphZoo.graphzoo.layers.hyp_layers import HyperbolicGraphConvolution as HGCNlayer
from GraphZoo.graphzoo.layers.hyp_layers import HNNLayer
from GraphZoo.graphzoo.layers.layers import GraphConvolution, Linear
from GraphZoo.graphzoo.layers.hyp_layers import HypLinear
from .layers import AttentionLayer, SelfAttention, MultiHeadSelfAttention, SELayer, CoGuidedLayer, InterViewAttention
from .loss_func import DGI_loss

from GraphZoo.graphzoo.manifolds import Euclidean, PoincareBall, Hyperboloid


class MManiST(nn.Module):

    def __init__(self, args):
        super(MManiST, self).__init__()

        self.args = args
        self.fusion_model = args.fusion_model

        self.eu_radius = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.po_radius = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.lo_radius = nn.Parameter(torch.tensor(1.), requires_grad=True)
        
        if not args.cuda == -1:
            self.eu_radius = nn.Parameter(self.eu_radius.to('cuda:0'))
            self.po_radius = nn.Parameter(self.po_radius.to('cuda:0'))
            self.lo_radius = nn.Parameter(self.lo_radius.to('cuda:0'))
        
        self.init_dim = args.init_dim

        self.eu_manifold = Euclidean()
        self.po_manifold = PoincareBall()
        self.lo_manifold = Hyperboloid()
        
        # define feature dims
        feat_dims =  args.feat_dims
        num_layer = len(feat_dims)-1

        if not args.act:
            act = lambda x:x
        else:
            act = getattr(F, args.act)
        acts = [act] * (num_layer)
        # acts[-1] = lambda x:x

        # define encoder sequential layer
        eu_en_layers = []
        for i in range(num_layer-1):
            manifold = self.eu_manifold
            in_dim = feat_dims[i]
            out_dim = feat_dims[i+1]
            # c_in = eu_curvatures[i]
            # c_out = eu_curvatures[i+1]
            r_in = self.eu_radius
            r_out = self.eu_radius
            act = acts[i]
            eu_en_layers.append(GraphConvolution(
                in_dim, out_dim, args.dropout, act, args.bias
            ))
        self.eu_en_seq = nn.Sequential(*eu_en_layers)

        po_en_layers = []
        for i in range(num_layer-1):
            manifold = self.po_manifold
            in_dim = feat_dims[i]
            out_dim = feat_dims[i+1]
            # c_in = po_curvatures[i]
            # c_out = po_curvatures[i+1]
            r_in = self.po_radius
            r_out = self.po_radius
            act = acts[i]
            po_en_layers.append(HGCNlayer(
                manifold, in_dim, out_dim, r_in, r_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
        self.po_en_seq = nn.Sequential(*po_en_layers)

        lo_en_layers = []
        for i in range(num_layer-1):
            manifold = self.lo_manifold
            in_dim = feat_dims[i]+1
            out_dim = feat_dims[i+1]+1
            # c_in = lo_curvatures[i]
            # c_out = lo_curvatures[i+1]
            r_in = self.lo_radius
            r_out = self.lo_radius
            act = acts[i]
            lo_en_layers.append(HGCNlayer(
                manifold, in_dim, out_dim, r_in, r_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
        self.lo_en_seq = nn.Sequential(*lo_en_layers)

        # define decoder sequential layer
        # define decoder sequential layer
        if self.fusion_model == 'concat':
            self.mid_layer = HNNLayer(
                self.eu_manifold, 3*feat_dims[num_layer], feat_dims[num_layer], self.eu_radius, args.dropout, act, args.bias
            )

        elif self.fusion_model == 'attention':
            self.mid_layer = AttentionLayer(feat_dims[num_layer])
            # self.de_seq = nn.Sequential(*de_layers)
        elif self.fusion_model == 'self-attention':
            self.mid_layer = SelfAttention(feat_dims[num_layer])
        elif self.fusion_model == 'multi-self-attention':
            self.mid_layer = MultiHeadSelfAttention(feat_dims[num_layer], 8)
        elif self.fusion_model == 'channel-attention':
            self.mid_layer = SELayer(3)
        elif self.fusion_model == 'co-guided':
            self.mid_layer = CoGuidedLayer(feat_dims[num_layer])
        elif self.fusion_model == 'interview-attention':
            self.mid_layer = InterViewAttention(feat_dims[num_layer])


        de_layers = []
        for i in range(num_layer):
            manifold = self.eu_manifold
            in_dim = feat_dims[num_layer-i]
            out_dim = feat_dims[num_layer-(i+1)]
            # c_in = eu_curvatures[num_layer-i]
            # c_out = eu_curvatures[num_layer-(i+1)]
            r_in = self.eu_radius
            r_out = self.eu_radius
            # c_in = curvatures[num_layer+i]
            # c_out = curvatures[num_layer+i+1]
            act = acts[i]

            de_layers.append(Linear(
                in_dim, out_dim, args.dropout, act, args.bias
            ))

        self.de_seq = nn.Sequential(*de_layers)

        # elif self.fusion_model == 'distillation':
        if True:
            eu_de_layers = []
            for i in range(num_layer):
                manifold = self.eu_manifold
                in_dim = feat_dims[num_layer-i]
                out_dim = feat_dims[num_layer-(i+1)]
                # c_in = eu_curvatures[num_layer-i]
                # c_out = eu_curvatures[num_layer-(i+1)]
                r_in = self.eu_radius
                r_out = self.eu_radius
                # c_in = curvatures[num_layer+i]
                # c_out = curvatures[num_layer+i+1]
                act = acts[i]

                eu_de_layers.append(Linear(
                    in_dim, out_dim, args.dropout, act, args.bias
                ))
            self.eu_de_seq = nn.Sequential(*eu_de_layers)

            po_de_layers = []
            for i in range(num_layer):
                manifold = self.eu_manifold
                in_dim = feat_dims[num_layer-i]
                out_dim = feat_dims[num_layer-(i+1)]
                # c_in = po_curvatures[num_layer-i]
                # c_out = po_curvatures[num_layer-(i+1)]
                r_in = self.po_radius
                r_out = self.po_radius
                # c_in = curvatures[num_layer+i]
                # c_out = curvatures[num_layer+i+1]
                act = acts[i]

                po_de_layers.append(HNNLayer(
                    manifold, in_dim, out_dim, r_in, args.dropout, act, args.bias
                ))
            self.po_de_seq = nn.Sequential(*po_de_layers)

            lo_de_layers = []
            for i in range(num_layer):
                manifold = self.eu_manifold
                in_dim = feat_dims[num_layer-i]
                out_dim = feat_dims[num_layer-(i+1)]
                # c_in = lo_curvatures[num_layer-i]
                # c_out = lo_curvatures[num_layer-(i+1)]
                r_in = self.lo_radius
                r_out = self.lo_radius
                # c_in = curvatures[num_layer+i]
                # c_out = curvatures[num_layer+i+1]
                act = acts[i]

                lo_de_layers.append(HNNLayer(
                    manifold, in_dim, out_dim, r_in, args.dropout, act, args.bias
                ))
            self.lo_de_seq = nn.Sequential(*lo_de_layers)

            # self.big_model = HNNLayer(
            #     self.eu_manifold, feat_dims[num_layer], feat_dims[num_layer], self.eu_radius, args.dropout, act, args.bias
            # )

            act = lambda x:x
            self.big_model = HGCNlayer(
                self.eu_manifold, feat_dims[num_layer-1], feat_dims[num_layer], self.eu_radius, self.eu_radius, args.dropout, act, args.bias, args.use_att, args.local_agg
            )

            self.DGI_loss = DGI_loss(feat_dims[num_layer])
    
    def po_curvature(self):
        return 1. / self.po_radius.pow(2)
    
    def lo_curvature(self):
        return 1. / self.lo_radius.pow(2)


    def encode(self, x, adj):

        # calculate loss function
        res = {}
        res['recon_loss_eu'] = self.recon_loss_eu(x, adj)
        res['recon_loss_po'] = self.recon_loss_po(x, adj)
        res['recon_loss_lo'] = self.recon_loss_lo(x, adj)
        res['recon_loss_combine'] = self.recon_loss_combine(x, adj)

        # res['cl_loss_1'] = self.cl_loss_1(x, adj)
        # res['cl_loss_2'] = self.cl_loss_2(x, adj)
        
        self.res = res

        return self.h

    def decode(self, h, adj):
        return self.recon_x

    def compute_metrics(self, embeddings, data, split):
        res = self.res

        metrics = {}
        
        output = self.recon_x
        return output, metrics

    def init_metric_dict(self):
        return {'loss':10000000}

    def has_improved(self, m1, m2):
        return True
        return m1["loss"] > m2["loss"]
    
    
    def recon_loss_eu(self, x, adj):

        x = F.dropout(x, self.args.dropout, training=True)

        self.x = x

        h_eu, _ = self.eu_en_seq.forward((x, adj))
        h_eu, _ = self.big_model((h_eu, adj))
        recon_x_eu = self.eu_de_seq(h_eu)

        loss_fn = nn.MSELoss()
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        return loss_fn(self.x, recon_x_eu)
    

    def recon_loss_po(self, x, adj):

        x = F.dropout(x, self.args.dropout, training=True)

        self.po_c = self.po_curvature()
        
        x_tan_po = self.po_manifold.proj_tan0(x, self.po_c)
        self.x_tan_po = x_tan_po

        x_hyp_po = self.po_manifold.expmap0(x_tan_po, c=self.po_c)
        x_hyp_po = self.po_manifold.proj(x_hyp_po, c=self.po_c)
        self.x_hyp_po = x_hyp_po

        h_po, _ = self.po_en_seq.forward((x_hyp_po, adj))
        h_po, _ = self.big_model((h_po, adj))

        recon_x_po = self.po_de_seq(h_po)

        loss_fn = nn.MSELoss()
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        return loss_fn(self.x, recon_x_po)
    
    
    def recon_loss_lo(self, x, adj):

        x = F.dropout(x, self.args.dropout, training=True)

        o = torch.zeros_like(x)
        x_lo = torch.cat([o[:, 0:1], x], dim=1)

        self.lo_c = self.lo_curvature()

        self.x = x
        x_tan_lo = self.lo_manifold.proj_tan0(x_lo, self.lo_c)
        self.x_tan_lo = x_tan_lo

        x_hyp_lo = self.lo_manifold.expmap0(x_tan_lo, c=self.lo_c)
        x_hyp_lo = self.lo_manifold.proj(x_hyp_lo, c=self.lo_c)
        self.x_hyp_lo = x_hyp_lo

        h_lo, _ = self.lo_en_seq.forward((x_hyp_lo, adj))

        # map back to tangent space
        h_lo = self.lo_manifold.proj_tan0(self.lo_manifold.logmap0(h_lo, c=self.lo_c), c=self.lo_c)[:, 1:]

        h_lo, _ = self.big_model((h_lo, adj))
        recon_x_lo = self.lo_de_seq(h_lo)
        
        loss_fn = nn.MSELoss()
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        return loss_fn(self.x, recon_x_lo)

    
    def recon_loss_combine(self, x, adj):

        x = F.dropout(x, self.args.dropout, training=True)

        o = torch.zeros_like(x)
        x_lo = torch.cat([o[:, 0:1], x], dim=1)

        self.po_c = self.po_curvature()
        self.lo_c = self.lo_curvature()

        self.x = x
        x_tan_po = self.po_manifold.proj_tan0(x, self.po_c)
        self.x_tan_po = x_tan_po
        x_tan_lo = self.lo_manifold.proj_tan0(x_lo, self.lo_c)
        self.x_tan_lo = x_tan_lo

        x_hyp_po = self.po_manifold.expmap0(x_tan_po, c=self.po_c)
        x_hyp_po = self.po_manifold.proj(x_hyp_po, c=self.po_c)
        self.x_hyp_po = x_hyp_po

        x_hyp_lo = self.lo_manifold.expmap0(x_tan_lo, c=self.lo_c)
        x_hyp_lo = self.lo_manifold.proj(x_hyp_lo, c=self.lo_c)
        self.x_hyp_lo = x_hyp_lo

        h_eu, _ = self.eu_en_seq.forward((x, adj))
        h_po, _ = self.po_en_seq.forward((x_hyp_po, adj))
        h_lo, _ = self.lo_en_seq.forward((x_hyp_lo, adj))

        # map back to tangent space
        h_po = self.po_manifold.proj_tan0(self.po_manifold.logmap0(h_po, c=self.po_c), c=self.po_c)
        h_lo = self.lo_manifold.proj_tan0(self.lo_manifold.logmap0(h_lo, c=self.lo_c), c=self.lo_c)[:, 1:]

        h_eu, _ = self.big_model((h_eu, adj))
        h_po, _ = self.big_model((h_po, adj))
        h_lo, _ = self.big_model((h_lo, adj))


        if self.fusion_model == 'concat':
            h = torch.cat([h_eu, h_po, h_lo], dim=1)
            # h = torch.cat([h_po, h_lo], dim=1)
            h = self.mid_layer(h)
        elif self.fusion_model == 'attention':
            h = torch.stack([h_eu, h_po, h_lo], dim=1)
            # h = torch.stack([h_po, h_lo], dim=1)
            h = self.mid_layer(h)
        elif self.fusion_model == 'self-attention':
            h = torch.stack([h_eu, h_po, h_lo], dim=1)
            h = self.mid_layer(h)
        elif self.fusion_model == 'multi-self-attention':
            h = torch.stack([h_eu, h_po, h_lo], dim=1)
            h = self.mid_layer(h)
        elif self.fusion_model == 'channel-attention':
            h = torch.stack([h_eu, h_po, h_lo], dim=1)
            h = self.mid_layer(h)
        elif self.fusion_model == 'co-guided':
            h = self.mid_layer(h_eu, h_po, h_lo)
        elif self.fusion_model == 'interview-attention':
            h = self.mid_layer(h_eu, h_po, h_lo, adj)

        # re normalize
        # h = (h - h.min()) / (h.max() - h.min())
        # h_po = (h_po - h_po.min()) / (h_po.max() - h_po.min())

        # h, _ = self.big_model((h, adj))
        self.h = h

        recon_x = self.de_seq(h)
        self.recon_x = recon_x
        
        loss_fn = nn.MSELoss()
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        return loss_fn(self.x, recon_x)
 
