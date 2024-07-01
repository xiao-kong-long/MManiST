import torch
import numpy as np

# def mask_node(features, mask_prob):
#     n_nodes = features.shape[0]
#     mask_rates = torch.FloatTensor(np.ones(n_nodes) * mask_prob)
#     masks = torch.bernoulli(1 - mask_rates)
#     mask_idx = masks.nonzero().squeeze(1)
#     return mask_idx

def mask_node(features, node_mask_prob=0.5, feature_mask_prob=0.5):
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    n_mask_nodes = int(node_mask_prob * n_nodes)
    n_mask_features = int(feature_mask_prob * n_features)

    node_perm = torch.randperm(n_nodes)
    mask_nodes = node_perm[: n_mask_nodes].unsqueeze(0)
    
    feature_perm = torch.randperm(n_features)
    mask_features = feature_perm[: n_mask_features]

    features[mask_nodes, mask_features] = 0.0

    return features   


