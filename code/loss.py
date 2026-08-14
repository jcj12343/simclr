import torch
import torch.nn.functional as F

def nt_xent_loss(z, temperature=0.5):
    """
    z: [2B, D] 拼接两个视图 view1+view2
    return: scalar loss
    """
    batch_2 = z.shape[0]
    B = batch_2 // 2
    # 计算相似度矩阵
    sim = torch.matmul(z, z.T) / temperature
    # 屏蔽自身对角线
    mask = torch.eye(batch_2, dtype=torch.bool, device=z.device)
    sim_no_self = sim[~mask].view(batch_2, batch_2 - 1)

    # 正确提取正样本对
    pos_list = []
    for i in range(B):
        pos_list.append(sim[i, i+B])
        pos_list.append(sim[i+B, i])
    pos_sim = torch.stack(pos_list).unsqueeze(1)

    # 屏蔽自身+正样本，剩余全部为负样本
    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask.fill_diagonal_(False)
    for i in range(B):
        neg_mask[i, i+B] = False
        neg_mask[i+B, i] = False
    neg_sim = sim[neg_mask].view(batch_2, batch_2 - 2)

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits, labels)
    return loss