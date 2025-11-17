import torch

# 抖动增强：添加高斯噪声
def jitter(x, sigma=0.8):
    noise = torch.randn_like(x) * sigma
    return x + noise

# 缩放增强：通道方向乘缩放因子
def scaling(x, sigma=1.1):
    factor = torch.normal(mean=2.0, std=sigma, size=(x.size(0), 1, x.size(2)), device=x.device)
    return x * factor

# 置换增强：随机打乱时间段顺序
def permutation(x, max_segments=5, seg_mode="random"):
    B, T, C = x.shape
    ret = torch.zeros_like(x)
    for i in range(B):
        num_segs = torch.randint(1, max_segments + 1, (1,)).item()
        if num_segs > 1:
            if seg_mode == "random":
                split_points = torch.randperm(T - 2, device=x.device)[:num_segs - 1] + 1
                split_points, _ = torch.sort(split_points)
                segments = torch.tensor_split(torch.arange(T, device=x.device), split_points.tolist())
            else:
                segments = torch.tensor_split(torch.arange(T, device=x.device), num_segs)
            permuted = torch.cat([seg[torch.randperm(len(seg))] for seg in segments])
            ret[i] = x[i][permuted]
        else:
            ret[i] = x[i]
    return ret

# 独热编码（PyTorch 实现）
def one_hot_encoding_torch(X, num_classes):
    return torch.nn.functional.one_hot(X, num_classes=num_classes).float()

# 主数据增强函数（仅 PyTorch 实现）
def DataTransform_T(data, model_params=None):
    aug_1 = jitter(data)
    aug_2 = scaling(data)
    aug_3 = permutation(data)

    B = data.shape[0]
    device = data.device

    # 为每个样本随机选择一种增强方式（0=jitter, 1=scaling, 2=permutation）
    li = torch.randint(0, 3, (B,), device=device)
    li_onehot = one_hot_encoding_torch(li, num_classes=3)

    # 选择对应增强结果
    aug_1 = aug_1 * li_onehot[:, 0].view(-1, 1, 1)
    aug_2 = aug_2 * li_onehot[:, 1].view(-1, 1, 1)
    aug_3 = aug_3 * li_onehot[:, 2].view(-1, 1, 1)

    aug_T = aug_1 + aug_2 + aug_3
    return data, aug_T
