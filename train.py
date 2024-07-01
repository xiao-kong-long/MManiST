import os
import argparse
import numpy as np
import pandas as pd
import torch

from run_MManiST import run_MManiST
from utils import save_image_clustering_all

tech = '10x Visium'
dataset = 'DLPFC'
section_id = '151507'
num_cluster = 7
ground_truth = True
n_epoch = 3500
method = 'MManiST'

if num_cluster == -1:
    if section_id in ['151669', '151670', '151671', '151672']:
        num_cluster = 5
    else:
        num_cluster = 7
else:
    num_cluster = num_cluster

print('---------------' + section_id + '------------------')

ARI, NMI = run_MManiST(tech=tech, dataset=dataset, section_id=section_id, num_cluster=num_cluster, ground_truth=ground_truth, n_epoch=n_epoch, exist_epoch=0, k_nei=5, scale=10)
torch.cuda.empty_cache()

save_image_clustering_all(method=method, tech=tech, dataset=dataset, ground_truth=ground_truth, n_epoch=n_epoch, section_id=section_id)

print('ARI: ', ARI)
print('NMI: ', NMI)