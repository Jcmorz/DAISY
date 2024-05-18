import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Projection, GCN, RGCN

class MODEST(nn.Module):
    def __init__(self, args, num_ents, num_rels, dis_num_in_gd):
        super(MODEST, self).__init__()
        self.g_encoder = GCN(nfeat=args.h_dim, nhid=args.h_dim) # g_data.x.shape[1]==17247, 修改了输入维度
        self.kg_encoder = RGCN(num_nodes=num_ents, nhid=args.h_dim, num_rels=num_rels*2)
        self.projection1 = Projection(args.h_dim, args.z_dim)
        self.projection2 = Projection(dis_num_in_gd, args.z_dim)
        self.mlp = Projection(args.z_dim, 1)
        self.celoss = nn.CrossEntropyLoss()
        self.activation_func = nn.LeakyReLU()
        self.tau = args.tau
        self.dis_path = args.data
        
    def forward(self, g_data, g_data_svd, kg_data, g2o, d2g, d_h2, index, similarity_true, lam_1, lam_2):
        kg_h = self.kg_encoder(kg_data)

        g_h = self.pooling(kg_h, g2o)
        g_data.x = g_h
        g_data_svd.x = g_h

        g_h1 = self.g_encoder(g_data)
        g_h2 = self.g_encoder(g_data_svd)

        g_h = 1/2 * (g_h1 + g_h2)

        d_h1 = self.pooling(g_h, d2g)
        d_embedding= self.projection1(d_h1)
        d_z2 = self.projection2(d_h2)

        d_z1 = self.activation_func(d_embedding[index.T[0]])
        d_z2 = self.activation_func(d_z2[index.T[1]])
        # d_z1 = d_embedding[index.T[0]]
        # d_z2 = d_z2[index.T[1]]

        d_z = 1 / 2 * (d_z1 + d_z2)

        similarity = torch.zeros(d_z.shape[0], d_z.shape[0])
        for i in range(d_z.shape[0]):
            for j in range(d_z.shape[0]):
                # similarity[i][j] = self.mlp(torch.exp(-torch.abs(d_z[i]-d_z[j])))
                similarity[i][j] = torch.dot(d_z[i], d_z[j])

        similarity = F.normalize(similarity)

        l_p = self.celoss(similarity + 1e-8, similarity_true)
        l_g = self.nce_loss(g_h1, g_h2)
        l_d = self.nce_loss(d_z1, d_z2)

        loss = l_p + lam_1 * l_g + lam_2 * l_d

        return d_z, d_embedding, similarity, loss

    def pooling(self, kg_h, y2x):
        y = torch.mm(y2x, kg_h)
        row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
        y = torch.div(y, row_sum)
        return y

    def nce_loss(self, x_1, x_2):
        x_1 = F.normalize(x_1, dim=1)
        x_2 = F.normalize(x_2, dim=1)
        similarity_matrix = x_1 @ x_2.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True) # 将矩阵上每一行的元素求和
        positives = torch.diagonal(similarity_matrix, 0)
        loss = -torch.log(positives * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss
    
    def get_similarity(self, g_data, g_data_svd, kg_data, g2o, d2g, d_h2, index):
        kg_h = self.kg_encoder(kg_data)

        g_h = self.pooling(kg_h, g2o)
        g_data.x = g_h
        g_data_svd.x = g_h

        g_h1 = self.g_encoder(g_data)
        g_h2 = self.g_encoder(g_data_svd)

        g_h = 1 / 2 * (g_h1 + g_h2)

        d_h1 = self.pooling(g_h, d2g)
        d_z1 = self.projection1(d_h1)
        d_z2 = self.projection2(d_h2)

        d_z1 = self.activation_func(d_z1[index.T[0]])
        d_z2 = self.activation_func(d_z2[index.T[1]])
        d_z = 1 / 2 * (d_z1 + d_z2)

        similarity = torch.zeros(d_z.shape[0], d_z.shape[0])
        for i in range(d_z.shape[0]):
            for j in range(d_z.shape[0]):
                # similarity[i][j] = self.mlp(torch.exp(-torch.abs(d_z[i]-d_z[j])))
                similarity[i][j] = torch.dot(d_z[i], d_z[j])

        return similarity
    
    def get_onto_embeddings(self, kg_data):
        return self.kg_encoder(kg_data)
