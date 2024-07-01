"""
Base model class
"""
import random
import math
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
from GraphZoo.graphzoo.utils.eval_utils import acc_f1
from GraphZoo.graphzoo.utils.loss_utils import sce_loss
from GraphZoo.graphzoo.utils.mask_utils import mask_node

from GraphZoo.graphzoo.layers.hyp_att_layers import GraphAttentionLayer as HGATlayer
from GraphZoo.graphzoo.layers.hyp_layers import HyperbolicGraphConvolution as HGCNlayer
from GraphZoo.graphzoo.layers.hyp_layers import HNNLayer
from GraphZoo.graphzoo.layers.layers import GraphConvolution as GCNlayer
from GraphZoo.graphzoo.layers.layers import GraphConvolution1 as GCNlayer1
from GraphZoo.graphzoo.layers.layers import GraphConvolution2, Linear2

from GraphZoo.graphzoo.manifolds import Euclidean, PoincareBall, Hyperboloid

EPS = 1e-15

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks

    Input Parameters
    ----------
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc] (type: str)'),
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN,HGAT] (type: str)'),
        'dim': (128, 'embedding dimension (type: int)'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall] (type: str)'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature (type: float)'),
        'r': (2.0, 'fermi-dirac decoder parameter for lp (type: float)'),
        't': (1.0, 'fermi-dirac decoder parameter for lp (type: float)'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification (type: str)'),
        'num-layers': (2, 'number of hidden layers in encoder (type: int)'),
        'bias': (1, 'whether to use bias (1) or not (0) (type: int)'),
        'act': ('relu', 'which activation function to use or None for no activation (type: str)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim (type: int)'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks (type: float)'),
        'use-att': (0, 'whether to use hyperbolic attention (1) or not (0) (type: int)'),
        'local-agg': (0, 'whether to local tangent space aggregation (1) or not (0) (type: int)')
        
    API Input Parameters
    ----------
        args: list of above defined input parameters from `graphzoo.config`
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to('cuda:' + str(args.cuda))
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
            
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
       
        self.weights = torch.Tensor([1.] * args.n_classes)
        
        if not args.cuda == -1:
            self.weights = self.weights.to('cuda:' + str(args.cuda))

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            nb_false_edges = len(data['train_edges_false'])
            nb_edges = len(data['train_edges'])
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, nb_false_edges, nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
   


    """
    hyperbolic graph contrastive learning neural network auto-encoder
    """

    def __init__(self, args):
        super(HGCLAutoEncoder, self).__init__()

        self.args = args
        
        self.po_manifold = PoincareBall()
        self.lo_manifold = Hyperboloid()
        self.eu_manifold = Euclidean()
        
        # define feature dims
        feat_dims =  args.feat_dims
        num_layer = len(feat_dims)-1
        poincare_curvatures = [nn.Parameter(torch.Tensor([1.]))] * (2*num_layer+1)
        lorentz_curvatures = [nn.Parameter(torch.Tensor([1.]))] * (2*num_layer+1)
        if not args.cuda == -1:
            for c in poincare_curvatures:
                c = c.to('cuda:' + str(args.cuda))
            for c in lorentz_curvatures:
                c = c.to('cuda:' + str(args.cuda))

        self.poincare_curvatures = poincare_curvatures
        self.lorentz_curvatures = lorentz_curvatures

        if not args.act:
            act = lambda x:x
        else:
            act = getattr(F, args.act)
        acts = [act] * (2*num_layer)

        # define encoder sequential layer
        poincare_en_layers = []
        lorentz_en_layers = []
        eucli_en_layers = []
        for i in range(num_layer):
            in_dim = feat_dims[i]
            out_dim = feat_dims[i+1]
            poincare_c_in = poincare_curvatures[i]
            poincare_c_out = poincare_curvatures[i+1]
            lorentz_c_in = lorentz_curvatures[i]
            lorentz_c_out = lorentz_curvatures[i+1]

            act = acts[i]
            poincare_en_layers.append(HGCNlayer(
                self.po_manifold, in_dim, out_dim, poincare_c_in, poincare_c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
            lorentz_en_layers.append(HGCNlayer(
                self.lo_manifold, in_dim+1, out_dim+1, lorentz_c_in, lorentz_c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
            eucli_en_layers.append(GCNlayer1(
                in_dim, out_dim, args.dropout, act, args.bias
            ))

        self.poincare_en_seq = nn.Sequential(*poincare_en_layers)
        self.lorentz_en_seq = nn.Sequential(*lorentz_en_layers)
        self.eucli_en_seq = nn.Sequential(*eucli_en_layers)

        # define decoder sequential layer
        poincare_de_layers = []
        lorentz_de_layers = []
        eucli_de_layers = []
        for i in range(num_layer):
            in_dim = feat_dims[num_layer-i]
            out_dim = feat_dims[num_layer-(i+1)]
            # poincare_c_in = poincare_curvatures[num_layer-i]
            # poincare_c_out = poincare_curvatures[num_layer-(i+1)]
            # lorentz_c_in = lorentz_curvatures[num_layer-i]
            # lorentz_c_out = lorentz_curvatures[num_layer-(i+1)]
            poincare_c_in = poincare_curvatures[num_layer+i]
            poincare_c_out = poincare_curvatures[num_layer+i+1]
            lorentz_c_in = lorentz_curvatures[num_layer+i]
            lorentz_c_out = lorentz_curvatures[num_layer+i+1]

            # act = acts[(num_layer-1)-i]
            act = acts[num_layer+i]

            poincare_de_layers.append(HGCNlayer(
                self.po_manifold, in_dim, out_dim, poincare_c_in, poincare_c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
            lorentz_de_layers.append(HGCNlayer(
                self.lo_manifold, in_dim+1, out_dim+1, lorentz_c_in, lorentz_c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
            ))
            eucli_de_layers.append(GCNlayer1(
                in_dim, out_dim, args.dropout, act, args.bias
            ))    
        
        self.poincare_de_seq = nn.Sequential(*poincare_de_layers)
        self.lorentz_de_seq = nn.Sequential(*lorentz_de_layers)
        self.eucli_de_seq = nn.Sequential(*eucli_de_layers)


    def encode(self, x, adj):

        # PoincareBall encode
        po_x_tan = self.po_manifold.proj_tan0(x, self.poincare_curvatures[0])
        self.po_x_tan = po_x_tan
        po_x_hyp = self.po_manifold.expmap0(po_x_tan, c=self.poincare_curvatures[0])
        po_x_hyp = self.po_manifold.proj(po_x_hyp, c=self.poincare_curvatures[0])
        po_h, _ = self.poincare_en_seq.forward((po_x_hyp, adj))

        # Lorentz encoder
        o = torch.zeros_like(x)
        x_ = torch.cat([o[:, 0:1], x], dim=1)
        lo_x_tan = self.lo_manifold.proj_tan0(x_, c=self.lorentz_curvatures[0])
        self.lo_x_tan = lo_x_tan
        lo_x_hyp = self.lo_manifold.proj_tan0(lo_x_tan, c=self.lorentz_curvatures[0])
        lo_h, _ = self.lorentz_en_seq.forward((lo_x_hyp, adj))

        # Eclidean encoder
        self.eu_x = x
        eu_h, _ = self.eucli_en_seq.forward((x, adj))

        # hyperbolic contrastive loss
        class HPCLoss(nn.Module):
            def __init__(self, args):
                self.args = args
                self.po_manifold = PoincareBall()
                self.lo_manifold = Hyperboloid()
                self.eu_manifold = Euclidean()
                super(HPCLoss, self).__init__()
            
            def forward(self, po_h, lo_h, eu_h, adj, po_c, lo_c):
                # intra loss
                loss = self.MI_intra(po_h, adj, self.po_manifold, po_c) + self.MI_intra(lo_h, adj, self.lo_manifold, lo_c) + self.MI_intra(eu_h, adj, self.eu_manifold, 0)

                # between poincare and lorentz   
                po_h_to_lo = self.po_manifold.to_hyperboloid(po_h, po_c)
                lo_h_to_po = self.lo_manifold.to_poincare(lo_h, lo_c)
                loss = loss + self.MI_inter(lo_h, po_h_to_lo, adj, self.lo_manifold, lo_c) + self.MI_inter(po_h, lo_h_to_po, adj, self.po_manifold, po_c)
                
                # between poincare and euclidean
                po_h_to_eu = self.po_manifold.proj_tan0(self.po_manifold.logmap0(po_h, c=po_c), c=po_c)
                eu_h_to_po = self.po_manifold.proj(self.po_manifold.expmap0(eu_h, c=po_c), c=po_c)
                loss = loss + self.MI_inter(eu_h, po_h_to_eu, adj, self.eu_manifold, 0) + self.MI_inter(po_h, eu_h_to_po, adj, self.po_manifold, po_c)
                
                # between lorentz and euclidean
                lo_h_to_eu = self.lo_manifold.proj_tan0(self.lo_manifold.logmap0(lo_h, c=lo_c), c=lo_c)[:,1:]
                o = torch.zeros_like(eu_h)
                eu_h_to_lo = self.lo_manifold.proj(self.lo_manifold.expmap0(eu_h, c=lo_c), c=lo_c)
                loss = loss + self.MI_inter(eu_h, lo_h_to_eu, adj, self.eu_manifold, 0) + self.MI_inter(lo_h[:,1:], eu_h_to_lo, adj, self.lo_manifold, lo_c)
                
                return loss

            def discriminate(self, p, q, manifold, c):
                dist = manifold.sqdist(p, q, c)
                return torch.clamp(dist, max=1.0)


            # def MI_inter(self, h_alpha_i, h_beta, i, manifold, c, m=10):
            #     pos_loss = -torch.log(
            #         self.discriminate(h_alpha_i, h_beta[i], manifold, c) + EPS
            #     )
            #     index_list = list(range(0, h_beta.shape[0]))
            #     index_list.remove(i)

            #     sample_index_list = np.random.choice(index_list, size=m, replace=False)
            #     neg_loss = 0
            #     for index in sample_index_list:
            #         neg_loss = neg_loss - torch.log(1 - 
            #             self.discriminate(h_alpha_i, h_beta[index], manifold, c) + EPS
            #         )
            #     neg_loss = neg_loss / m
                
            #     return pos_loss + neg_loss

            # def negtive_sample(self, edge, n):
            #     edge_ = torch.empty(2,0)
            #     c_arr = list(range(0,n))
            #     random.shuffle(c_arr)
            #     recode_r = -1
            #     recode_c_list = []
            #     for r, c in edge:
            #         if recode_r == -1:
            #             recode_r = r
            #             recode_c_list.append(c)
            #         elif r != recode_r:
            #             if c_arr[recode_r] not in recode_c_list:
            #                 add_row = torch.tensor([recode_r, c_arr[recode_r]]).unsqueeze(1)
            #                 edge_ = torch.cat((edge_, add_row), dim=1)
            #             recode_r = r
            #             recode_c_list = []
            #         else:
            #             recode_c_list.append(c)
            #     return edge_.type(torch.long)

            def negtive_sample(self, n):
                c_arr = np.array(range(0,n))
                node1 = torch.tensor(c_arr).unsqueeze(0)
                random.shuffle(c_arr)
                node2 = torch.tensor(c_arr).unsqueeze(0)
                edge_ = torch.cat((node1, node2), dim=0)

                return edge_.type(torch.long)
            
            def two_order_negtive_sample(self, adj):
                # sample two order neibohood as negtive sample
                
                adj_two_order = torch.mm(torch.mm(adj, adj), adj)
                one = torch.ones_like(adj_two_order)
                # replace all value that is greater then 0 to 1
                adj_two_order = torch.where(adj_two_order>0, one, adj_two_order)
                # remove all terms in one-order adj
                adj_two_order = adj_two_order - adj
                assert (adj>=0).all()
                edge_ = torch.nonzero(adj_two_order)
                # print(edge_.shape)
                return edge_

            def remove_self_loop(self, edge):
                edge_ = edge[edge[:,0]==edge[:,1]]
                return edge_

            def MI_intra(self, h_alpha, adj, manifold, c, m=10):
                edge = torch.nonzero(adj)
                pos_loss = -torch.log(
                    self.discriminate(h_alpha[edge[0,:],:], h_alpha[edge[1,:],:], manifold, c) + EPS
                ).mean()

                # edge_ = self.negtive_sample(h_alpha.shape[0])
                edge_ = self.two_order_negtive_sample(adj)
                neg_loss = -torch.log(1 - 
                    self.discriminate(h_alpha[edge_[0,:],:], h_alpha[edge_[1,:],:], manifold, c) +  EPS
                ).mean()

                return pos_loss + neg_loss
            
            def MI_inter(self, h_alpha, h_beta, adj, manifold, c, m=10):
                edge = torch.nonzero(adj)
                edge_ = self.remove_self_loop(edge)
                pos_loss = -torch.log(
                    self.discriminate(h_alpha, h_beta, manifold, c) + EPS
                ).mean()
                neg_loss = -torch.log(1 - 
                    self.discriminate(h_alpha[edge_[0,:],:], h_beta[edge_[1,:],:], manifold, c) + EPS
                ).mean()

                return pos_loss + neg_loss

        loss_fn = HPCLoss(self.args)
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        self.contra_loss = loss_fn(po_h, lo_h, eu_h, adj, self.poincare_curvatures[0], self.lorentz_curvatures[0])


        # convert to eclidean space
        po_c = self.poincare_curvatures[0]
        lo_c = self.lorentz_curvatures[0]
        # po_h_to_eu = self.po_manifold.proj_tan0(self.po_manifold.logmap0(po_h, c=po_c), c=po_c)
        # lo_h_to_eu = self.lo_manifold.proj_tan0(self.lo_manifold.logmap0(lo_h, c=lo_c), c=lo_c)  

        return (po_h, lo_h, eu_h)
        # return po_h
        

    def decode(self, h, adj):
        po_h, lo_h, eu_h = h
        # po_h = h
        
        # PoincareBall decoder
        po_recon_x, _ = self.poincare_de_seq.forward((po_h, adj))
        po_recon_x = self.po_manifold.proj_tan0(self.po_manifold.logmap0(po_recon_x, c=self.poincare_curvatures[0]), c=self.poincare_curvatures[0])
        # Lorentz decoder
        lo_recon_x, _ = self.lorentz_de_seq.forward((lo_h, adj))
        lo_recon_x = self.lo_manifold.proj_tan0(self.lo_manifold.logmap0(lo_recon_x, c=self.lorentz_curvatures[0]), c=self.lorentz_curvatures[0])
        # Euclidean decoder
        eu_recon_x, _ = self.eucli_de_seq.forward((eu_h, adj))

        
        return (po_recon_x, lo_recon_x, eu_recon_x)
        # return po_recon_x

    def compute_metrics(self, embeddings, data, split):
        
        metrics = {'loss': self.contra_loss}
        return metrics

        po_recon_x, lo_recon_x, eu_recon_x = self.decode(embeddings, data['adj_train_norm'])
        # po_recon_x = self.decode(embeddings, data['adj_train_norm'])
        loss_fn = nn.MSELoss()
        if not self.args.cuda == -1:
            loss_fn = loss_fn.to('cuda:' + str(self.args.cuda))

        # reconstrutiong loss function
        loss = loss_fn(po_recon_x, self.po_x_tan) + loss_fn(lo_recon_x, self.lo_x_tan) + loss_fn(eu_recon_x, self.eu_x) + self.contra_loss
        # loss = loss_fn(po_recon_x, self.po_x_tan)

        metrics = {'loss': loss}
        return (po_recon_x, lo_recon_x, eu_recon_x), metrics
        # return po_recon_x, metrics

    def init_metric_dict(self):
        return {'loss':10000000}

    def has_improved(self, m1, m2):
        return m1["loss"] > m2["loss"]