import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder




def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, class_num, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.pool_size = 512

        #Cluster-level projection layer
        self.cluster_projector_t = nn.Sequential(
            # nn.Linear(output_dims*1, 256),
            nn.Linear(output_dims * self.pool_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, class_num),
            nn.Softmax(dim=1)
        )

        # #更复杂的mlp
        # self.cluster_projector_t = nn.Sequential(
        #     # 将隐藏层维度依次连接
        #     nn.Linear(output_dims * self.pool_size, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, class_num),
        #     nn.Softmax(dim=1)
        # )



        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co


        x_pool = F.adaptive_max_pool1d(
            x.transpose(1, 2),
            output_size=self.pool_size
        ).transpose(1, 2)  # x_pool: B x 4 x output_dims


        h_x = x_pool.reshape(x_pool.shape[0], -1)
        x_c = self.cluster_projector_t(h_x)

        #L2归一化
        x = F.normalize(x, p=2, dim=-1)
        return x, x_c


    # def forward_cluster(self, x_in_t):
    #     #After passing through the representation network, cluster-level mapping of the time series yields clustering results
    #     x = self.time_encoder.forward(x_in_t) #输出：Tensor:(batch_size, output_size)
    #     h_time = x.reshape(x.shape[0], -1)
    #     z_time = self.cluster_projector_t(h_time)
    #     c=torch.argmax(z_time,dim=1)
    #     return c