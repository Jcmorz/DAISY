import time
import pickle
import logging
import os.path as osp
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from evaluation import get_metrics, get_metrics_cg, get_ground_truth, topk


torch.manual_seed(0)

def pooling(x, y2x):
    x = torch.mm(y2x, x)
    row_sum = torch.sum(y2x, dim=1).clamp(min=1e-8).reshape(-1, 1)
    x = torch.div(x, row_sum)
    return x

    
class Trainer(object):
    def __init__(self, model, optimizer, log_every_n_steps, device, dis_path):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        self.dis_path = dis_path
        self.writer = SummaryWriter("./runs/final")
        self.linear = torch.nn.Linear(32, 1)
        logging.basicConfig(filename=osp.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        
    def load_data(self, g_data, g_data_svd, kg_data, g2o, d2g, d_h2, lam_1, lam_2):
        self.g_data = g_data.to(self.device)
        self.g_data_svd = g_data_svd.to(self.device)
        self.kg_data = kg_data.to(self.device)
        self.g2o = g2o.to(self.device)
        self.d2g = d2g.to(self.device)
        self.d_h2 = d_h2
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        
    def nce_loss(self, gz, kgz, labels):
        gz = F.normalize(gz, dim=1)
        kgz = F.normalize(kgz, dim=1)
        similarity_matrix = gz @ kgz.T
        similarity_matrix = torch.exp(similarity_matrix / self.tau)
        similarity_matrix_sum = torch.sum(similarity_matrix, 1, keepdim=True) # 将矩阵上每一行的元素求和
        positives_sum = torch.sum(similarity_matrix * labels, 1)
        loss = -torch.log(positives_sum * (similarity_matrix_sum**(-1)) + 1e-8).mean()
        return loss
    
    def train(self, epochs):
        time_begin = time.time()
        print(f"Start JCLModel training for {epochs} epochs.")
        logging.info(f"Start JCLModel training for {epochs} epochs.")
        logging.info(f"Training device: {self.device}.")
        training_range = tqdm(range(epochs))

        y_true1, y_true2, y_true = get_ground_truth(self.dis_path)
        index_mx = np.array([np.array(y_true1, dtype=int).flatten(), np.array(y_true2, dtype=int).flatten()])
        self.index = np.unique(index_mx.T, axis=0)

        self.similarity_true = torch.zeros(self.index.shape[0], self.index.shape[0])
        for i in y_true:
            self.similarity_true[i] = 1
            self.similarity_true[i[1], i[0]] = 1

        true_tmp = torch.mm(self.similarity_true, self.similarity_true)
        true_tmp = torch.sign(true_tmp - torch.diag_embed(torch.diag(true_tmp)))
        self.similarity_true = torch.sign(self.similarity_true + true_tmp)

        auc_final = 0.0
        aupr_final = 0.0
        cov_final = 0.0
        auc_cg_final = 0.0
        aupr_cg_final = 0.0

        import os
        save_dir = "visualization_images"
        os.makedirs(save_dir, exist_ok=True)


        # turn to training mode
        self.model.train()
        for epoch in training_range:
            d_z, d_embedding, similarity, loss = self.model(self.g_data, self.g_data_svd, self.kg_data, self.g2o, self.d2g,
                                                            self.d_h2, self.index, self.similarity_true, self.lam_1, self.lam_2)



            # 使用t-SNE对数据进行降维
            tsne = TSNE(n_components=2, perplexity=10)  # 设置输出维度为2
            z_visual = d_embedding.detach().numpy()
            data_tsne = tsne.fit_transform(z_visual)

            # 可视化降维后的数据
            plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
            plt.title(f"t-SNE Visualization (Epoch {epoch + 1})")

            # 保存可视化结果为图片
            save_path = os.path.join(save_dir, f"tsne_visualization_epoch{epoch + 1}.png")
            plt.savefig(save_path)
            plt.close()


            training_range.set_description('Loss %.4f' % loss.item())   # 设置进度条左边显示的信息

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.log_every_n_steps == 0:
                with torch.no_grad():
                    auc, aupr, cov = get_metrics(similarity, self.similarity_true)
                    auc_cg, aupr_cg = get_metrics_cg(d_embedding, self.dis_path)
                    self.writer.add_scalar('loss', loss, global_step=epoch)
                    self.writer.add_scalar('auc', auc, global_step=epoch)
                    self.writer.add_scalar('aupr', aupr, global_step=epoch)
                    self.writer.add_scalar('cov', cov, global_step=epoch)
                    self.writer.add_scalar('auc_cg', auc_cg, global_step=epoch)
                    self.writer.add_scalar('aupr_cg', aupr_cg, global_step=epoch)
                    if auc > auc_final:
                        auc_final = auc
                    if aupr > aupr_final:
                        aupr_final = aupr
                    if cov > cov_final:
                        cov_final = cov
                    if auc_cg > auc_cg_final:
                        auc_cg_final = auc_cg
                    if aupr_cg > aupr_cg_final:
                        aupr_cg_final = aupr_cg
                logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tAUROC: {auc}\tAUPR: {aupr}\tCOV: {cov}\tAUROC_CG: {auc_cg}\tAUPR_CG: {aupr_cg}")

        time_end = time.time()
        logging.info("Training has finished.")
        logging.info(f"Training takes {(time_end-time_begin)/60} mins")
        checkpoint_name = "checkpoint_{:03d}.pth.tar".format(epochs)
        torch.save({'epoch':epochs,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict()},
                   f=osp.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata have been saved at {self.writer.log_dir}.")
        pickle.dump(d_z,
                    open(osp.join(self.writer.log_dir, "disease_embedding.pkl"), "wb"))
        logging.info(f"Disease embedding has been saved at {self.writer.log_dir}.")
        auc, aupr, cov = self.infer()
        logging.info(f"AUROC: {auc} \tAUPR: {aupr} \tCOV: {cov}\tAUROC_CG: {auc_cg}\tAUPR_CG: {aupr_cg}")
        logging.info(f"AUROC_FINAL: {auc_final} \tAUPR_FINAL: {aupr_final} \tCOV_FINAL: {cov_final}\tAUROC_CG_F: {auc_cg_final}\tAUPR_CG_F: {aupr_cg_final}")
        print(f"AUROC_FINAL: {auc_final} \tAUPR_FINAL: {aupr_final} \tCOV_FINAL: {cov_final}\tAUROC_CG_F: {auc_cg_final}\tAUPR_CG_F: {aupr_cg_final}")

    def infer(self):
        with torch.no_grad():
            self.model.eval()
            similarity = self.model.get_similarity(self.g_data, self.g_data_svd, self.kg_data, self.g2o, self.d2g,
                                                   self.d_h2, self.index)
            auc, aupr, cov = get_metrics(similarity, self.similarity_true)

            # a, tk = topk(d, self.dis_path, 10) # topk()的函数有点问题，暂时先把这一行注释掉了。

            print(f"AUROC: {auc*100} | AUPR: {aupr*100} | COV: {cov}")
        # return auroc, ap, a, tk
        return auc, aupr, cov
