import numpy as np
import pandas as pd
import torch
import os
from GraphZoo.graphzoo.manifolds.poincare import PoincareBall
from GraphZoo.graphzoo.manifolds.hyperboloid import Hyperboloid
from GraphZoo.graphzoo.optimizers import RiemannianAdam
# from GraphZoo.graphzoo.trainers import Trainer
from MManiST.train import Trainer
from GraphZoo.graphzoo.config import parser

from MManiST.model import MManiST

from HGCNAE_utils import load_ST_graph_data
from utils import refine_label, mclust_R

def train_MManiST(tech='10x Visium', dataset='DLPFC', section_id='151676', n_epoch=1000, exist_epoch=0, lr=0.001, gradient_clipping=5., 
                weight_decay=0.0001,verbose=True, random_seed=42, save_loss=False, 
                device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),k_nei=3, scale=3,
                after_cluster=False, num_cluster=-1):
    
    seed  = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    args = parser.parse_args()
    data = load_ST_graph_data(tech=tech, dataset=dataset, section_id=section_id, k_nei=k_nei, scale=scale, after_cluster=after_cluster)

    args.seed = seed

    # parameter setting
    
    args.act = 'elu'
    if device == 'cpu':
        args.cuda = -1
    else:
        args.cuda = 0
    args.init_dim = data['features'].shape[1]
    args.feat_dims = [args.init_dim, 128, 30]
    args.recon = True
    # args.use_att = 1

    args.lr = lr
    args.weight_decay = weight_decay
    args.epochs = n_epoch
    args.exist_epoch = exist_epoch
    args.dropout = 0
    
    args.method = 'MManiST'
    args.tech = tech
    args.dataset = dataset
    args.section_id = section_id
    args.num_cluster = num_cluster

    args.fusion_model = 'interview-attention'

    args.manifold = 'Other'

    model = MManiST(args) # model would to cuda in trainnign file
    # lr = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                                    betas=args.betas, eps=args.eps, amsgrad=args.amsgrad,
                                    )
    

    trainer = Trainer(args, model, optimizer, data)
    z, out = trainer.run()
    
    z = z.to('cpu').detach().numpy()
    out = out.to('cpu').detach().numpy()

    return z, out

from utils import calculate_ari, calculate_nmi
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression, PLSSVD

def run_MManiST(tech='10x Visium', dataset='DLPFC', section_id='151676', num_cluster=-1, ground_truth=False, n_epoch=100, exist_epoch=0, k_nei=3, scale=3):
    
    z, out = train_MManiST(tech=tech, dataset=dataset, section_id=section_id, n_epoch=n_epoch, exist_epoch=exist_epoch, device='cuda', k_nei=k_nei, scale=scale, num_cluster=num_cluster)

    save_dir = os.path.join('output', tech, dataset, section_id, 'MManiST')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    embedding_url = os.path.join(save_dir, 'embedding'+str(n_epoch)+'.csv')
    embedding_df = pd.DataFrame(z)
    embedding_df.to_csv(embedding_url, index=False, header=False)

    embedding_pca = z

    # do clustering
    clustering_res = mclust_R(embedding_pca, num_cluster=num_cluster)
    clustering_df = pd.DataFrame(clustering_res, dtype='category')
    clustering_url = os.path.join(save_dir, 'clustering_res.csv')
    clustering_df.to_csv(clustering_url)
    clustering_res = refine_label(method='MManiST', tech=tech, dataset=dataset, section_id=section_id, radius=40)
    ARI =  calculate_ari(tech, dataset, section_id, 'MManiST')
    NMI = calculate_nmi(tech, dataset, section_id, 'MManiST')
    
    return ARI, NMI

    
    
    
