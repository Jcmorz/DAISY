import os

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn.functional as F

from sklearn import metrics
from xlrd import open_workbook


root = os.path.dirname(os.path.dirname(__file__))
path_dis2id = root + "/data/dis2id.txt"
path_ent2id = root + "/data/entity2id.txt"
path_pos = root + "/data/positive.xlsx"
path_pos_id = root + "/data/positive_id.xlsx"
path_neg1 = root + "/data/random1.xlsx"
path_neg2 = root + "/data/random2.xlsx"

########################################################################
# Disease Similarity Evaluation
########################################################################


def load_d2g(path):
    return sp.load_npz(path).todense()

# load dis2id
def load_dmap(path):
    dmap = dict()
    inv = dict()
    with open(path) as f:
        f.readline()
        for line in f:
            dis, id = line.strip().split()
            dmap[dis] = (int)(id)
            inv[(int)(id)] = dis
    return dmap, inv

def load_disid(path):
    idmap = dict()
    with open(path, "r") as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            disid, name = line[0], line[1]
            if disid not in idmap.keys():
                idmap[disid] = name
            else:
                continue
    return idmap

# load /data/.xlsx
def load_xlsx(path, dmap):
    data = open_workbook(path)
    table = data.sheets()[0]
    dataset = []
    for i in range(table.nrows):
        dis1 = table.cell(i, 1).value
        dis2 = table.cell(i, 3).value
        try:
            d1 = (int)(dmap[dis1])
            d2 = (int)(dmap[dis2])
            dataset.append((d1, d2))
        except:
            continue

    return dataset


def load_xlsx_rewrite(path1, path2, dmap1, dmap2):
    data1 = open_workbook(path1)
    data2 = open_workbook(path2)
    table1 = data1.sheets()[0]
    table2 = data2.sheets()[0]
    dataset1 = []
    dataset2 = []
    dataset3 = []
    for i in range(table1.nrows):
        dis1 = table1.cell(i, 1).value
        dis2 = table1.cell(i, 3).value
        try:
            d1 = (int)(dmap1[dis1])
            d2 = (int)(dmap1[dis2])
            d3 = (int)(dmap2[dis1])
            d4 = (int)(dmap2[dis2])
            dataset1.append((d1, d2))
            dataset2.append((d3, d4))
        except:
            continue

    for i in range(table2.nrows):
        index1 = (int)(table2.cell(i, 1).value)
        index2 = (int)(table2.cell(i, 3).value)
        dataset3.append((index1, index2))

    return dataset1, dataset2, dataset3


def get_ground_truth(dis_path):
    dmap1, _ = load_dmap(dis_path+"/dis2id.txt")
    dmap2, _ = load_dmap(dis_path + "/signed_dis2id.tsv")
    y_true1, y_true2, y_true = load_xlsx_rewrite(path_pos, path_pos_id, dmap1, dmap2)
    return y_true1, y_true2, y_true


def get_metrics_cg(dis_embeddings, dis_path, plot=False):
    """
    Calculate disease similarity AUC metrics
    :param dis_sim: disease embeddings matrix
    :param plot: whether to draw ROC
    :return: AUROC, AP score
    """
    dmap, _ = load_dmap(dis_path+"/dis2id.txt")
    pos_data = load_xlsx(path_pos, dmap)
    neg_data1 = load_xlsx(path_neg1, dmap)
    neg_data2 = load_xlsx(path_neg2, dmap)
    data = pos_data + neg_data1 + neg_data2
    data = np.asarray(data)

    y = np.zeros(len(data), dtype=np.int32)
    for i in range(len(pos_data)):
        y[i] = 1

    a = F.normalize(dis_embeddings[data[:, 0]])
    b = F.normalize(dis_embeddings[data[:, 1]])
    x1 = F.cosine_similarity(a, b).view(-1).numpy()
    x = torch.diag(torch.mm(a, b.T)).numpy()

    auroc = metrics.roc_auc_score(y, x)
    ap = metrics.average_precision_score(y, x)

    return auroc, ap


def get_metrics(similarity, similarity_true):
    """
    Calculate disease similarity prediction evaluation metrics
    """
    y_true = torch.flatten(similarity_true)
    y_score = torch.flatten(similarity)

    auc = metrics.roc_auc_score(y_true, y_score)
    aupr = metrics.average_precision_score(y_true, y_score)
    cov = torch.std(y_score)/torch.mean(y_score)
    return auc, aupr, cov

def topk(dis_embeddings, dis_path, k):
    """ Find topk similar diseases for each anchor. """
    dmap, inv = load_dmap(dis_path+"/dis2id.txt")
    idmap = load_disid(root+"/data/raw/disease_mappings_to_attributes.tsv")
    pos_data = load_xlsx(path_pos, dmap)
    anchor = set(list(zip(*pos_data))[0])
    anchor_index = torch.tensor([i for i in anchor])
    
    emb_norm = F.normalize(dis_embeddings, dim=1)
    res = torch.mm(emb_norm, emb_norm.T).fill_diagonal_(0.)
    _, topk_index = res[anchor_index].topk(k)
    anchor = [idmap[inv[i]] for i in anchor_index.tolist()]
    top = [[idmap[inv[k]] for k in j] for j in topk_index.tolist()]
    return anchor, top