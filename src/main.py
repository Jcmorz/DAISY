import argparse

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from utils import *
from model import MODEST
from sg_encoder import process_query
from trainer import Trainer


parser = argparse.ArgumentParser(description="PyTorch JCLModel")
parser.add_argument("--data", default="../data", help="path to dataset")
parser.add_argument("--h_dim", default=16, type=int, help="dimension of layer h")
parser.add_argument("--z_dim", default=16, type=int, help="dimension of layer z")
parser.add_argument("--tau", default=0.8, type=float, help="softmax temperature")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--decay", default=1e-8, type=float, help="weight decay")
parser.add_argument("--epochs", default=2, type=int, help="train epochs") # default=200
parser.add_argument("--disable-cuda", default=True, action="store_true", help="disable CUDA")
parser.add_argument("--log-every-n-steps", default=1, type=int, help="log every n steps")
parser.add_argument("--lam_1", default=0.5, type=float, help="scale control hyper-parameter")
parser.add_argument("--lam_2", default=0.5, type=float, help="scale control hyper-parameter")

parser.add_argument("--input_path", default="../data/signed_dis.tsv")
parser.add_argument("--output_path", default="../data/scores.tsv")
parser.add_argument("--output_type", default="rd", help="Choose from the following: rp, rn, rd, both")

args = parser.parse_args()
print(args)
device = torch.device("cuda" if not args.disable_cuda and torch.cuda.is_available() else "cpu")

# Load knowledge graph statistics
with open(args.data+"/entity2id.txt", "r") as f:
    num_ents = (int)(f.readline())  # num_ents==43987
with open(args.data+"/relation2id.txt", "r") as f:
    num_rels = (int)(f.readline())

# Load GO knowledge graph for RGCN Model
train_triples = load_triples(args.data) # GO graph: 90185 edges.
edge_index, edge_type = get_kg_data(train_triples, num_rels) # 把有向图转为双向图, 180370 edges
kg_data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_ents)

# Load human net for GCN Model
hnadj = load_sparse(args.data+"/hnet.npz") # gene graph: 17247*17247
src = hnadj.row
dst = hnadj.col
hn_edge_weight = torch.tensor(np.hstack((hnadj.data, hnadj.data)), dtype=torch.float)
hn_edge_weight = (hn_edge_weight - hn_edge_weight.min()) / (hn_edge_weight.max() - hn_edge_weight.min())
hn_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)

# Load gene2GO align
g2o = load_sparse(args.data+"/g2o.npz") # gene*ontology: 17247*43987
g2o = mx_to_torch_sparse_tesnsor(g2o).to_dense()

x = generate_sparse_one_hot(g2o.shape[0]) # x: 17247*17247
g_data = Data(x=x, edge_index=hn_edge_index, edge_weight=hn_edge_weight)

# SVD
u, s, v = torch.svd_lowrank(torch.tensor(hnadj.todense()), q = 5)
hnadj_svd = u @ torch.diag(s) @ v.T
del hnadj, u, v # 内存不够，所以重构的hnadj_svd暂时丢弃了原本边上的权重，后续应当考虑是否有边权值为负的情况出现
# hnadj_svd = sp.coo_matrix(np.mat(hnadj_svd.numpy()))
# hn_svd_edge_weight = torch.tensor(np.hstack((hnadj_svd.data, hnadj_svd.data)), dtype=torch.float)
# hn_svd_edge_weight = (hn_svd_edge_weight - hn_svd_edge_weight.min()) / (hn_svd_edge_weight.max() - hn_svd_edge_weight.min())
hn_svd_edge_index = torch.tensor(np.vstack((np.concatenate([src, dst]), np.concatenate([dst, src]))), dtype=torch.long)
# g_data_svd = Data(x=x, edge_index=hn_svd_edge_index, edge_weight=hn_svd_edge_weight)
g_data_svd = Data(x=x, edge_index=hn_svd_edge_index)

d2g = load_sparse(args.data+"/d2g.npz") # disease*gene: 30170*17247
d2g = mx_to_torch_sparse_tesnsor(d2g).to_dense()

d_h2 = process_query(args.input_path, args.output_path, args.output_type, c=0.15, epsilon=1e-9, beta=0.5, gamma=0.5, max_iters=300, handles_deadend=True)
d_h2 = torch.tensor(d_h2, dtype=torch.float32)
# d_h2 = torch.randn(264, 264)

model = MODEST(args, num_ents, num_rels, np.size(d_h2, 0)) # np.size(d_h2, 0) == 264, ground_truth中所有的疾病个数
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

trainer = Trainer(model, optimizer=optimizer, log_every_n_steps=args.log_every_n_steps, device=device, dis_path=args.data)
trainer.load_data(g_data, g_data_svd, kg_data, g2o, d2g, d_h2, args.lam_1, args.lam_2)

print("Finish initializing...")
print("---------------------------------------")
trainer.train(args.epochs)

def test(path):
    checkpoint = torch.load(path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    auroc, aupr, cov =trainer.infer()

    # with open("./results/topid.csv", "w") as f:
    #     for i in range(len(a)):
    #         f.write(str(a[i])+","+",".join(str(j) for j in tk[i])+"\n")

test("./runs/final/checkpoint_005.pth.tar")
