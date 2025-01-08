import torch


def get_local_score(q_reps, p_reps, all_scores):
    """获取 queries 和 passages 的局部得分。

    Args:
        q_reps (torch.Tensor): queries 的表示。
        p_reps (torch.Tensor): passages 的表示。
        all_scores (torch.Tensor): 计算得到的所有 query-passage 的得分。

    Returns:
        torch.Tensor: 用于计算损失的局部得分。
    """
    group_size = p_reps.size(0) // q_reps.size(0) # 每个 query 对应的 passages 数量
    indices = torch.arange(0, q_reps.size(0), device=q_reps.device) * group_size # 每个 query 在 all_scores 中的索引
    specific_scores = []
    for i in range(group_size):
        # 从 all_scores 中提取每个 query 对应的第 i 个 passage 的得分
        specific_scores.append(
            all_scores[torch.arange(q_reps.size(0), device=q_reps.device), indices + i] # (batch_size, group_size)
        )
    # 将所有特定得分堆叠在一起，并调整形状为 (batch_size, group_size)
    return torch.stack(specific_scores, dim=1).view(q_reps.size(0), -1)





def _compute_similarity(q_reps, p_reps):
    """使用内积计算 query 和 passage 表示之间的相似度。

    Args:
        q_reps (torch.Tensor): queries 的表示。
        p_reps (torch.Tensor): passages 的表示。

    Returns:
        torch.Tensor: 计算得到的相似度矩阵。
    """
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))


def compute_score(q_reps, p_reps, temperature):
    """计算 queries 和 passages 之间的得分。

    Args:
        q_reps (torch.Tensor): queries 的表示。
        p_reps (torch.Tensor): passages 的表示。
        temperature (float): 温度参数，用于调整得分。

    Returns:
        torch.Tensor: 调整后的得分。
    """
    scores = _compute_similarity(q_reps, p_reps) / temperature  # (batch_size, group_size)
    scores = scores.view(q_reps.size(0), -1)  # (batch_size, group_size)
    return scores


def compute_local_score(q_reps, p_reps, temperature):
    """计算 queries 和 passages 的局部得分。

    Args:
        q_reps (torch.Tensor): queries 的表示。
        p_reps (torch.Tensor): passages 的表示。
        temperature (float): 温度参数，用于调整得分。

    Returns:
        torch.Tensor: 用于计算损失的局部得分。
    """
    all_scores = compute_score(q_reps, p_reps, temperature)

    loacl_scores = get_local_score(q_reps, p_reps, all_scores)
    return loacl_scores



def compute_loss(scores, target):
    """使用交叉熵计算损失。

    Args:
        scores (torch.Tensor): 计算得到的得分。
        target (torch.Tensor): 目标值。

    Returns:
        torch.Tensor: 计算得到的交叉熵损失。
    """
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    return cross_entropy(scores, target)


def compute_no_in_batch_neg_loss(q_reps, p_reps, temperature):
    """
    在不使用批内负样本和跨设备负样本的情况下计算损失。
    
    Args:
        q_reps (torch.Tensor): queries 的表示，形状为 (batch_size, dim)。
        p_reps (torch.Tensor): passages 的表示，形状为 (batch_size * group_size, dim)。
        temperature (float): 温度参数，用于调整得分。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 返回局部得分和计算得到的损失。
    """
    local_scores = compute_local_score(q_reps, p_reps, temperature)   # (batch_size, group_size)
    local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)  # (batch_size)
    
    loss = compute_loss(local_scores, local_targets)

    return local_scores, loss