# toolbox of hyperbolic graph convolutional neural network auto-encoder

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
import scipy.sparse as sp
import scipy.stats
import torch

from sklearn.decomposition import PCA

from utils import read_anndata


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_ST_graph_data(tech='10x Visium', dataset='DLPFC', section_id='151676', k_nei=3, scale=3, after_cluster=False, method='HGCNAE'):

    adata = read_anndata(tech=tech, dataset=dataset, section_id=section_id, ground_truth=True)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable']]

    sc.pp.normalize_total(adata, target_sum=1)

    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # sc.pp.normalize_total(adata, target_sum=1)
    
    num_cell = adata.X.shape[0]
    adata.obs.index = range(num_cell) # change the index to number
    

    # build graph
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    model = 'KNN'
    # model = 'Radius'
    rad_cutoff = 150
    k_cutoff = k_nei

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            # KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
            KNN_list.append(pd.DataFrame(zip(indices[it,:],[it]*indices.shape[1], distances[it,:])))

    if model == 'KDTree':
        tree = sklearn.neighbors.KDTree(coor, leaf_size=2)
        distances, indices = tree.query(coor, k=k_cutoff+1)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    
    if model == 'BallTree':
        tree = sklearn.neighbors.BallTree(coor, leaf_size=2)
        distances, indices = tree.query(coor, k=k_cutoff+1)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    
    edge_df = pd.concat(KNN_list)
    edge_df.columns = ['Cell1', 'Cell2', 'Distance']
    edge_df_ = edge_df.loc[edge_df['Distance']>0]

    row_list = edge_df_['Cell1'].values.tolist()
    col_list = edge_df_['Cell2'].values.tolist()
    dist_list = [1] * len(row_list)

    adj_mat = sp.csr_matrix((np.ones(len(row_list)), (row_list, col_list)), shape=(num_cell, num_cell))

    # 利用余弦相似度+knn构图
    from sklearn.metrics.pairwise import cosine_distances
    distances_arr = cosine_distances(adata.X)

    x = adata.X.todense()
    x = np.array(x)
    # similarity_arr = scipy.stats.spearmanr(x.T).correlation
    # similarity_arr = np.absolute(similarity_arr)
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=5, metric='precomputed').fit(distances_arr)
    distances, indices = nbrs.kneighbors(adata.X)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    edge_df = pd.concat(KNN_list)
    edge_df.columns = ['Cell1', 'Cell2', 'Distance']
    edge_df_ = edge_df.loc[edge_df['Distance']>0]
    
    row_list = edge_df_['Cell1'].values.tolist()
    col_list = edge_df_['Cell2'].values.tolist()
    dist_list = edge_df_['Distance'].values.tolist()

    ones = np.ones_like(dist_list)

    adj_mat_feat = sp.csr_matrix((ones, (row_list, col_list)), shape=(num_cell, num_cell))



    # adj_normalized = normalize_adj(adj_mat + np.eye(adj_mat.shape[0])) + np.zeros(shape=(num_cell, num_cell))
    adj_normalized = normalize_adj(adj_mat) + np.eye(adj_mat.shape[0])
    # adj_normalized = adj_mat + np.zeros(shape=(num_cell, num_cell))
    adj_normalized_feat = normalize_adj(adj_mat_feat) + np.eye(adj_mat_feat.shape[0])

    # print(type(adj_normalized))
    num_edge = np.count_nonzero(adj_normalized)
    edge_cell_ratio = float(num_edge) / num_cell
    print('num of cell: ', num_cell)
    print('num of edge: ', num_edge)
    print('raito edge on cell: ', edge_cell_ratio)
    

    features_tensor = scipy_sparse_mat_to_torch_sparse_tensor(adata.X).to_dense()
    # features_tensor = torch.ones(num_cell, 3000)

    # dim reduction
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=128, random_state=42)
    # features = pca.fit_transform(adata.X.todense())
    # features_tensor = torch.FloatTensor(features)


    # adj_mat_tensor = scipy_sparse_mat_to_torch_sparse_tensor(adj_mat).to_dense()
    # adj_mat_tensor = scipy_sparse_mat_to_torch_sparse_tensor(adj_normalized).to_dense()
    adj_mat_tensor = torch.FloatTensor(adj_normalized)
    adj_mat_tensor_feat = torch.FloatTensor(adj_normalized_feat)
    
    # adj_mat_tensor_feat = None

    data = {'adj_train_norm':adj_mat_tensor, 'features':features_tensor, 'adj_train_norm_feat':adj_mat_tensor_feat}

    return data



def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return sp.csr_matrix(adj)
