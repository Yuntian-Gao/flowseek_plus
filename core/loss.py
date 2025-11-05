# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ⭐️ 新增: 导入 torch_scatter
from torch_scatter import scatter_mean


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

# ⭐️ 新增: 系数一致性损失辅助函数
import torch
from torch_scatter import scatter_mean

def get_consistency_loss(coeffs, mask_8x):
 
    # mask_8x (N, 1, H, W) -> (N, H, W) 并转为 long 类型
    mask = mask_8x.squeeze(1).long()
    N, H, W = mask.shape
    
    # coeffs (N, C, H, W)
    N_c, C, H_c, W_c = coeffs.shape
    
    
    assert H_c == H and W_c == W and N_c == N, "Coeffs and mask dimensions do not match!"

   
    coeffs_flat = coeffs.view(N, C, H * W)
    # 展平掩码并扩展
    # mask_flat_idx 包含负数, 我们用它来创建 *最终* 的 valid_mask
    mask_flat_idx = mask.view(N, 1, H * W).expand_as(coeffs_flat) # (N, C, H*W)

   
    mask_flat_idx_safe = mask_flat_idx + 1

    
    mean_coeffs_per_id = scatter_mean(src=coeffs_flat, index=mask_flat_idx_safe, dim=2)

   
    mean_coeffs_gathered = torch.gather(mean_coeffs_per_id, dim=2, index=mask_flat_idx_safe)
    

 
    l1_loss = torch.abs(coeffs_flat - mean_coeffs_gathered)

   
    valid_mask = (mask_flat_idx >= 0)

    
    if valid_mask.sum() > 0:
       
        consistency_loss = l1_loss[valid_mask].mean()
    else:
       
        consistency_loss = torch.tensor(0.0, device=coeffs.device)
    
    return consistency_loss


# ⭐️ 修改: sequence_loss
def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, consistency_weight=0.1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    
    # ⭐️ 新增: 初始化一致性损失
    coeff_loss = 0.0

    # ⭐️ 新增: 从 output 中获取 mask 和 coeffs
    mask_8x = output['mask_8x']
    coeffs_list = output['coeffs']

    # 检查预测数量是否匹配
    assert n_predictions == len(coeffs_list), "Number of flow and coefficient predictions must be equal"

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        
        # --- 1. 原始光流损失 (nf_loss) ---
        loss_i = output['nf'][i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        
        # 确保 final_mask 中至少有一个 True 值，以避免除以零
        if final_mask.sum() > 0:
            flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        else:
            # 如果没有有效像素，则此迭代的损失为 0
            flow_loss += 0.0

        # --- 2. 新的系数一致性损失 ---
        # coeffs_list[i] 是 (N, 8, H/8, W/8)
        # mask_8x 是 (N, 1, H/8, W/8)
        current_coeffs = coeffs_list[i]
        consistency_loss_i = get_consistency_loss(current_coeffs, mask_8x)
        
        coeff_loss += i_weight * consistency_loss_i

    
    total_loss = flow_loss+ consistency_weight * coeff_loss
    
   
    return total_loss,consistency_weight * coeff_loss