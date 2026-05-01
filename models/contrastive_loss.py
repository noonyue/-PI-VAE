"""
Contrastive Learning Loss Functions
对比学习损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比学习损失函数

    核心思想：
    - 拉近同类样本（正样本对）的潜变量表示
    - 推远不同类样本（负样本对）的潜变量表示

    适用场景：
    - 提升小样本泛化能力
    - 增强噪声鲁棒性
    - 学习更具判别性的特征表示
    """
    def __init__(self, temperature=0.07, use_cosine_similarity=True):
        """
        Args:
            temperature: 温度参数，控制分布的平滑度（越小越陡峭）
            use_cosine_similarity: 是否使用余弦相似度（推荐）
        """
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, z_i, z_j, labels):
        """
        计算InfoNCE损失

        Args:
            z_i: 第一个视图的潜变量 (batch, latent_dim)
            z_j: 第二个视图的潜变量 (batch, latent_dim)
            labels: 样本标签 (batch,)

        Returns:
            loss: InfoNCE损失标量
        """
        batch_size = z_i.size(0)

        # 归一化（用于余弦相似度）
        if self.use_cosine_similarity:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        # 计算相似度矩阵
        # sim_matrix[i, j] = similarity(z_i[i], z_j[j])
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature  # (batch, batch)

        # 构建正样本mask（同类样本为正样本对）
        labels = labels.contiguous().view(-1, 1)
        pos_mask = (labels == labels.T).float()  # (batch, batch)

        # 移除对角线（自己与自己不算正样本对）
        pos_mask = pos_mask - torch.eye(batch_size, device=pos_mask.device)

        # 计算InfoNCE损失
        # log(exp(sim(i,j)) / sum_k(exp(sim(i,k))))
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 只对正样本对计算损失
        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1.0)  # 避免除零

        loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
        loss = loss.mean()

        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (监督对比学习损失)

    相比InfoNCE，SupCon考虑了所有正样本对，而不仅仅是增强视图对
    适合有标签的场景

    Reference: Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: (batch, latent_dim) 或 (batch, n_views, latent_dim)
            labels: (batch,)

        Returns:
            loss: SupCon损失标量
        """
        device = features.device

        # 如果是多视图，展平
        if len(features.shape) == 3:
            batch_size, n_views, latent_dim = features.shape
            features = features.view(batch_size * n_views, latent_dim)
            labels = labels.repeat(n_views)
        else:
            batch_size = features.size(0)

        # 归一化
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 构建mask
        labels = labels.contiguous().view(-1, 1)
        mask = (labels == labels.T).float().to(device)

        # 移除对角线
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # 计算损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class HardNegativeMining:
    """
    Hard Negative Mining（困难负样本挖掘）

    选择与锚点样本相似度最高的负样本，提升对比学习效果
    """
    @staticmethod
    def mine_hard_negatives(z_anchor, z_candidates, labels_anchor, labels_candidates, top_k=5):
        """
        挖掘困难负样本

        Args:
            z_anchor: 锚点样本潜变量 (n_anchor, latent_dim)
            z_candidates: 候选样本潜变量 (n_candidates, latent_dim)
            labels_anchor: 锚点样本标签 (n_anchor,)
            labels_candidates: 候选样本标签 (n_candidates,)
            top_k: 每个锚点选择的困难负样本数量

        Returns:
            hard_negatives: (n_anchor, top_k, latent_dim)
            hard_negative_labels: (n_anchor, top_k)
        """
        # 归一化
        z_anchor = F.normalize(z_anchor, dim=1)
        z_candidates = F.normalize(z_candidates, dim=1)

        # 计算相似度
        similarity = torch.matmul(z_anchor, z_candidates.T)  # (n_anchor, n_candidates)

        # 构建负样本mask
        labels_anchor = labels_anchor.view(-1, 1)
        labels_candidates = labels_candidates.view(1, -1)
        negative_mask = (labels_anchor != labels_candidates).float()

        # 将正样本的相似度设为-inf
        similarity = similarity * negative_mask + (1 - negative_mask) * (-1e9)

        # 选择top-k困难负样本
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)

        # 提取困难负样本
        hard_negatives = z_candidates[top_k_indices]  # (n_anchor, top_k, latent_dim)
        hard_negative_labels = labels_candidates[0, top_k_indices]  # (n_anchor, top_k)

        return hard_negatives, hard_negative_labels


class ContrastiveLossWithHardNegatives(nn.Module):
    """
    结合困难负样本挖掘的对比学习损失
    """
    def __init__(self, temperature=0.07, hard_negative_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(self, z_i, z_j, labels, hard_negatives=None):
        """
        Args:
            z_i: 第一个视图 (batch, latent_dim)
            z_j: 第二个视图 (batch, latent_dim)
            labels: 标签 (batch,)
            hard_negatives: 困难负样本 (batch, k, latent_dim)

        Returns:
            loss: 对比学习损失
        """
        batch_size = z_i.size(0)

        # 归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 正样本对相似度
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # (batch,)

        # 负样本相似度
        neg_sim = torch.matmul(z_i, z_j.T) / self.temperature  # (batch, batch)

        # 移除对角线（正样本对）
        mask = torch.eye(batch_size, device=z_i.device).bool()
        neg_sim = neg_sim.masked_fill(mask, -1e9)

        # 如果有困难负样本，增加权重
        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, dim=2)
            hard_sim = torch.matmul(z_i.unsqueeze(1), hard_negatives.transpose(1, 2)).squeeze(1)
            hard_sim = hard_sim / self.temperature * self.hard_negative_weight
            neg_sim = torch.cat([neg_sim, hard_sim], dim=1)

        # 计算损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels_contrastive = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
        loss = F.cross_entropy(logits, labels_contrastive)

        return loss


# 测试代码
if __name__ == "__main__":
    print("测试对比学习损失函数...")

    # 创建测试数据
    batch_size = 32
    latent_dim = 64
    n_classes = 9

    # 模拟两个增强视图的潜变量
    z_i = torch.randn(batch_size, latent_dim)
    z_j = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, n_classes, (batch_size,))

    # 测试InfoNCE损失
    print("\n1. 测试InfoNCE损失")
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss = infonce_loss(z_i, z_j, labels)
    print(f"   InfoNCE Loss: {loss.item():.4f}")

    # 测试SupCon损失
    print("\n2. 测试SupCon损失")
    supcon_loss = SupConLoss(temperature=0.07)
    loss = supcon_loss(z_i, labels)
    print(f"   SupCon Loss: {loss.item():.4f}")

    # 测试困难负样本挖掘
    print("\n3. 测试困难负样本挖掘")
    z_anchor = torch.randn(10, latent_dim)
    z_candidates = torch.randn(50, latent_dim)
    labels_anchor = torch.randint(0, n_classes, (10,))
    labels_candidates = torch.randint(0, n_classes, (50,))

    hard_negatives, hard_labels = HardNegativeMining.mine_hard_negatives(
        z_anchor, z_candidates, labels_anchor, labels_candidates, top_k=5
    )
    print(f"   Hard negatives shape: {hard_negatives.shape}")
    print(f"   Hard negative labels shape: {hard_labels.shape}")

    # 测试带困难负样本的对比学习损失
    print("\n4. 测试带困难负样本的对比学习损失")
    contrastive_loss = ContrastiveLossWithHardNegatives(temperature=0.07)
    loss = contrastive_loss(z_i, z_j, labels, hard_negatives=None)
    print(f"   Contrastive Loss (without hard negatives): {loss.item():.4f}")

    # 为每个样本生成困难负样本
    hard_negs = torch.randn(batch_size, 5, latent_dim)
    loss_with_hard = contrastive_loss(z_i, z_j, labels, hard_negatives=hard_negs)
    print(f"   Contrastive Loss (with hard negatives): {loss_with_hard.item():.4f}")

    print("\n✓ 对比学习损失函数测试通过！")
