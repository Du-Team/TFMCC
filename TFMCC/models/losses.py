import torch
from torch import nn
import torch.nn.functional as F
import math

def hierarchical_contrastive_loss(z1, z2, i_temperature, t_temperature,alpha=0.5,  temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss +=  alpha * instance_contrastive_loss(z1, z2, i_temperature)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, t_temperature)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, i_temperature)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2, temperature=1.0):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B

    # 应用温度系数
    sim = sim / temperature  # 将相似度矩阵除以温度系数

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2, temperature=1.0):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T

    # 应用温度系数
    sim = sim / temperature  # 将相似度矩阵除以温度系数

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


#用于簇级别的对比学习，目标是通过计算簇之间的相似度来使每个簇内的样本具有相似的特征表示，同时通过交叉熵约束优化簇的分布。
class ClusterLoss(nn.Module):
    """
    Cluster-level loss
    """
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        #温度系数 τ 用来缩放相似度矩阵，使得模型可以控制正样本和负样本之间的差异。
        #较小的温度系数会使相似度差异更大，导致更强的区分效果，较大的温度系数则会使得差异更小，从而有更多的容忍度。
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0) #掩码矩阵的对角线元素设置为 0
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask #

    def forward(self, c_i, c_j):
        #计算交叉熵约束
        p_i = c_i.sum(0).view(-1) #计算簇 c_i 中每个类别的总和
        p_i /= p_i.sum() #每个簇被取到的概率
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()#math.log(p_i.size(0))公式中没有
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j #交叉熵约束

        c_i = c_i.t()#[B, K]-> [K， B]  #转置后每列代表一个样本的特征表示
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0) #[2K, B]

        #c.unsqueeze(1)：[2K, B]-> [2K, 1, B]
        #c.unsqueeze(0)：[2K, B]-> [1, 2K, B]
        # nn.CosineSimilarity(dim=2)：余弦相似度计算时传入向量会广播为[2K, 2K, B](每个样本对 i, j 都有一个形状为 [B] 的特征向量)，
        #结果是[2k, 2k] 其中每个元素表示两个样本之间的相似度
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        # 使用内积相似度而不是余弦相似度:
        # sim = torch.matmul(c, c.t()) / self.temperature  # 计算内积相似度，结果 shape 为 [2K, 2K]

        sim_i_j = torch.diag(sim, self.class_num)#右上四分之一的矩阵的主对角线
        sim_j_i = torch.diag(sim, -self.class_num)#左下四分之一的矩阵的主对角线

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #[2k, 1]
        negative_clusters = sim[self.mask].reshape(N, -1)#[2k, 2k-2] (N=2k) #mask去除自对比（簇和簇 主对角线），和正样本（簇和簇的增强 右上四分之一和左下四分之一的矩阵的主对角线）

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1) #(N, 1 + (2K-2))#少了一个对应公式里的i != j

        #logits：模型输出的未经 softmax 处理的原始分数（形状为 (N, 1 + (2K-2))）。
        #标签：真实的类别标签，全0  [2k, 1]
        loss = self.criterion(logits, labels) #
        loss /= N
        # print(f"loss_c:{loss}, ne_loss:{ne_loss}")
        return loss + ne_loss

