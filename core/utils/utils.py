# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
def assign_uniqueId(segmentation: torch.Tensor) -> torch.Tensor:
    """
    将单通道分割图映射为唯一的实体 ID，跨样本保证 ID 连续。
    
    参数：
    - segmentation (torch.Tensor): 形状为 (N, 1, H, W) 的张量，像素值范围 [0, 255]。
    
    返回：
    - torch.Tensor: 形状为 (N, H, W) 的张量，每个像素值对应唯一实体 ID。背景像素为 -1。
    """
    N, H, W = segmentation.shape
    device = segmentation.device

    # 将输入 reshape 为 (N, H*W)
    segmentation_flat = segmentation.view(N, H * W)

    # 计算跨样本偏移量（每个样本有 256 个可能的 ID）
    offset = torch.arange(N, device=device) * 256  # (N,)
    adjusted_ids = segmentation_flat + offset.unsqueeze(1)  # (N, H*W)

    # 展平所有像素的 adjusted_id
    adjusted_ids_flat = adjusted_ids.view(-1)  # (N*H*W,)

    # 找到所有唯一 adjusted_id 及其逆索引
    unique_ids, inverse_indices = torch.unique(
        adjusted_ids_flat, sorted=True, return_inverse=True
    )

    # 初始化 ID 映射（从 1 开始）
    unique_id_map = torch.arange(1, unique_ids.size(0) + 1, device=device, dtype=torch.int32)

    # 判断背景像素（原始像素值为 0 )
    original_values = unique_ids % 256  # 还原原始像素值
    is_background = (original_values == 0)
    unique_id_map[is_background] = -1  # 背景 ID 设为 -1

    # 映射回最终的实体 ID
    entity_ids_flat = unique_id_map[inverse_indices]  # (N*H*W,)
    entity_ids = entity_ids_flat.view(N,H, W)

    return entity_ids
def forward_flow_to_backward(flow_forward: torch.Tensor) -> torch.Tensor:
    """
    从前向光流 (1 -> 2) 近似构造反向光流 (2 -> 1)。
    
    使用 "soft forward splatting" (软前向散布) 方法：
    1. 帧1中的每个像素 p1 = (x1, y1) 移动到 p2 = (x1 + flow_fwd(p1), y1 + flow_fwd(p1))。
    2. 对应的反向光流定义在 p2 处，其值为 flow_bwd(p2) = -flow_fwd(p1)。
    3. 由于 p2 是浮点坐标，我们将 -flow_fwd(p1) 的值通过双线性插值权重 "散布" (splat) 
       到 p2 周围的四个整数像素上。
    4. 我们使用两个张量：一个累加加权后的光流值 (flow_backward_sum)，
       一个累加权重 (weights_sum)。
    5. 最终的反向光流 = flow_backward_sum / (weights_sum + epsilon)，
       这正确地处理了冲突（加权平均）和孔洞（保持为0）。
    """
    N, _, H, W = flow_forward.shape
    device = flow_forward.device

    # 1. 创建 p1 坐标网格 (x1, y1)
    # [H, W]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    # [N, H, W]
    x1 = xx.float().expand(N, -1, -1)
    y1 = yy.float().expand(N, -1, -1)

    # 2. 计算 p2 目标坐标 (x2, y2)
    dx = flow_forward[:, 0, :, :]  # [N, H, W]
    dy = flow_forward[:, 1, :, :]
    x2 = x1 + dx
    y2 = y1 + dy

    # 3. 获取要散布的值 V = -flow_fwd(p1)
    v_dx = -dx  # [N, H, W]
    v_dy = -dy  # [N, H, W]
    
    # 将 v_dx 和 v_dy 堆叠，以便后续处理
    # [N, 2, H, W]
    v_flow = torch.stack([v_dx, v_dy], dim=1)

    # 4. 初始化输出累加器
    flow_backward_sum = torch.zeros(N, 2, H, W, device=device)
    weights_sum = torch.zeros(N, 1, H, W, device=device)

    # 5. 计算 p2 的四个最近邻整数坐标
    x2_floor = torch.floor(x2).long()
    y2_floor = torch.floor(y2).long()
    x2_ceil = x2_floor + 1
    y2_ceil = y2_floor + 1

    # 6. 计算双线性散布权重
    # (与 grid_sample 的采样权重相反)
    w_tl = (x2_ceil.float() - x2) * (y2_ceil.float() - y2) # top-left
    w_tr = (x2 - x2_floor.float()) * (y2_ceil.float() - y2) # top-right
    w_bl = (x2_ceil.float() - x2) * (y2 - y2_floor.float()) # bottom-left
    w_br = (x2 - x2_floor.float()) * (y2 - y2_floor.float()) # bottom-right

    # 7. 定义一个辅助函数来执行 scatter_add_
    # 这个函数将把 (N, H, W) 的源值散布到 (N, C, H, W) 的目标画布上
    def splat(y_coord, x_coord, weight, v_flow_to_splat):
        """
        参数:
        y_coord, x_coord: [N, H, W] 目标 y/x 坐标 (整数)
        weight: [N, H, W] 散布权重
        v_flow_to_splat: [N, 2, H, W] 要散布的光流值
        """
        
        # --- 安全性检查：创建有效性掩码 ---
        mask = (x_coord >= 0) & (x_coord < W) & (y_coord >= 0) & (y_coord < H)
        weight_masked = weight * mask.float()
        
        # --- 准备 scatter_add_ 的索引 ---
        y_clamped = torch.clamp(y_coord, 0, H - 1)
        x_clamped = torch.clamp(x_coord, 0, W - 1)
        
        # [N, H, W]
        flat_idx = (y_clamped * W + x_clamped).long()
        
        # --- 准备 scatter_add_ 的源 (src) ---
        w_src = weight_masked.unsqueeze(1)
        v_src = v_flow_to_splat * w_src  # 预先乘以权重
        
        # --- 执行 scatter_add_ ---
        # [N, 1, H*W]
        flat_idx_weights = flat_idx.unsqueeze(1).view(N, 1, -1)
        
        # [N, 2, H*W]
        # vvvvvvvvvvvvvv 这是修正行 vvvvvvvvvvvvvvv
        flat_idx_flow = flat_idx.unsqueeze(1).expand(-1, 2, -1, -1).view(N, 2, -1)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        weights_sum.view(N, 1, -1).scatter_add_(
            dim=2, 
            index=flat_idx_weights, 
            src=w_src.view(N, 1, -1)
        )
        flow_backward_sum.view(N, 2, -1).scatter_add_(
            dim=2, 
            index=flat_idx_flow, 
            src=v_src.view(N, 2, -1)
        )

    # 8. 向四个角进行散布
    splat(y2_floor, x2_floor, w_tl, v_flow) # Top-Left
    splat(y2_floor, x2_ceil,  w_tr, v_flow) # Top-Right
    splat(y2_ceil,  x2_floor, w_bl, v_flow) # Bottom-Left
    splat(y2_ceil,  x2_ceil,  w_br, v_flow) # Bottom-Right

    # 9. 归一化：除以总权重
    # 添加一个小的 epsilon 以避免除以零（处理孔洞）
    flow_backward = flow_backward_sum / (weights_sum + 1e-8)

    return flow_backward
def warp_mask_to_mask2_hw(
    mask1_HW: torch.Tensor, 
    flow1_backward_HW: torch.Tensor, 
    scale: float
) -> torch.Tensor:
    """
    使用 *反向* 光流 (2 -> 1) 来扭曲 mask1_HW，生成 *下采样* 的 mask2_hw。

    Args:
        mask1_HW (torch.Tensor): 形状为 [N, 1, H, W] 的源掩码 (ID)。
        flow1_backward_HW (torch.Tensor): 形状为 [N, 2, H, W] 的 *反向* 光流 (2 -> 1)。
                                          (x_src, y_src) = (x_dest + dx, y_dest + dy)
                                          dx, dy 是在 (H, W) 像素空间的偏移量。
        scale (float): 生成的 mask2_hw 相对于 mask1_HW 的缩放比例 (例如, 0.5)。

    Returns:
        torch.Tensor: 形状为 [N, 1, h, w] 的扭曲后的掩码 (mask2_hw)。
                      h = round(H * scale), w = round(W * scale)
    """
    N, _, H, W = mask1_HW.shape
    device = mask1_HW.device

    # 1. 计算目标 (h, w) 维度
    h = int(H * scale)
    w = int(W * scale)

    if h < 1 or w < 1:
        raise ValueError(f"Target scale {scale} results in invalid dimensions: ({h}, {w})")

    # 2. 创建目标网格 grid_dest_hw [N, h, w, 2]
    # 这个网格代表了 (h, w) 输出图像中的 [-1, 1] 坐标 (x, y 顺序)
    yy_hw, xx_hw = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    # grid_dest_hw: [h, w, 2] (x, y)
    grid_dest_hw = torch.stack((xx_hw, yy_hw), dim=2) 
    # grid_dest_hw: [N, h, w, 2]
    grid_dest_hw = grid_dest_hw.unsqueeze(0).repeat(N, 1, 1, 1)

    # 3. 在 (h, w) 网格点上采样 (H, W) 光流场
    # 我们使用 grid_dest_hw (它在 [-1, 1] 范围内) 
    # 来采样 flow1_backward_HW (它定义在 (H, W) 上)
    # 得到在 (h, w) 位置上的插值光流
    flow1_backward_hw = F.grid_sample(
        flow1_backward_HW, 
        grid_dest_hw,
        mode='bilinear', # 光流是连续的，用双线性插值
        padding_mode='border',
        align_corners=True 
    )
    # flow1_backward_hw: [N, 2, h, w]

    # 4. 将采样后的 (h, w) 光流转换为 [N, h, w, 2] (dx, dy)
    flow_perm_hw = flow1_backward_hw.permute(0, 2, 3, 1)

    # 5. 归一化光流
    # flow_perm_hw 包含的 (dx, dy) 仍然是 (H, W) 空间的像素偏移量
    # 我们需要将它们归一化到 [-1, 1] 范围，以便与 grid_dest_hw 相加
    # 归一化是相对于 *源* 空间 (H, W) 的
    flow_norm_hw = torch.zeros_like(flow_perm_hw)
    if W > 1:
        flow_norm_hw[..., 0] = flow_perm_hw[..., 0] * (2.0 / (W - 1)) # dx -> x_norm
    if H > 1:
        flow_norm_hw[..., 1] = flow_perm_hw[..., 1] * (2.0 / (H - 1)) # dy -> y_norm

    # 6. 计算源网格 (Source Grid)
    # grid_src: [N, h, w, 2]
    # 对于 (h, w) 中的每个点，它在 [-1, 1] 源空间 (H, W) 中的对应采样位置
    grid_src = grid_dest_hw + flow_norm_hw 

    # 7. 采样 mask1_HW，生成 mask2_hw
    mask1_float = mask1_HW.float()
    
    mask2_warped_hw = F.grid_sample(
        mask1_float, 
        grid_src, 
        mode='nearest', # ID 掩码必须用 'nearest'
        padding_mode='border', 
        align_corners=True
    )
    # mask2_warped_hw: [N, 1, h, w]
    
    return mask2_warped_hw.long()



def sample_correlation_from_masks_with_neighborhood_fallback(
    corr: torch.Tensor, 
    mask1_HW: torch.Tensor, 
    mask2_hw: torch.Tensor, 
    flow1_forward_HW: torch.Tensor, 
    m: int,
    r: float
) -> torch.Tensor:
    """
    根据 mask1 (高分辨率) 和 mask2 (低分辨率) 采样代价体 corr。
    
    如果ID匹配失败，则在 flow1_forward (高分辨率) 预测的、并缩放至
    低分辨率空间的邻域 (半径 r) 内采样。

    Args:
        corr (torch.Tensor): 代价体, 形状 [NHW, 1, hw]
        mask1_HW (torch.Tensor): 图像1的掩码 (高分辨率), 形状 [N, 1, H, W]
        mask2_hw (torch.Tensor): 图像2的掩码 (低分辨率), 形状 [N, 1, h, w]
        flow1_forward_HW (torch.Tensor): *正向* 光流 (1 -> 2), 
                                         形状 [N, 2, H, W],
                                         光流值(dx, dy)定义在 (H, W) 坐标系中。
                                         通道 0: dx, 通道 1: dy
        m (int): 每个点要采样的数量
        r (float): 邻域回退采样的半径 (在 (h, w) 空间中定义)

    Returns:
        torch.Tensor: 稀疏化的代价体, 形状 [NHW, 1, m]
    """
    # --- 1. 形状提取 ---
    N, _, H, W = mask1_HW.shape
    _, _, h, w = mask2_hw.shape
    
    HW = H * W
    hw = h * w
    NHW = N * H * W
    device = corr.device

    # --- 2. 形状验证 (修正了原始代码中的hw/HW混淆) ---
    corr = corr.view(NHW, 1, hw) # 确保形状正确
    if corr.shape[0] != NHW or corr.shape[2] != hw:
        raise ValueError(f"Corr shape {corr.shape} is incompatible. "
                         f"Expected [NHW, 1, hw] = [{NHW}, 1, {hw}]")
    if flow1_forward_HW.shape != (N, 2, H, W):
        raise ValueError(f"flow1_forward_HW shape {flow1_forward_HW.shape} is incorrect. "
                         f"Expected [N, 2, H, W] = [{N}, 2, {H}, {W}]")
    if mask2_hw.shape != (N, 1, h, w):
         raise ValueError(f"mask2_hw shape {mask2_hw.shape} is incorrect. "
                         f"Expected [N, 1, h, w] = [{N}, 1, {h}, {w}]")

    # --- 3. 尺度因子 ---
    # (H, W) 坐标 -> (h, w) 坐标
    scale_h = h / H
    scale_w = w / W

    # --- 4. 掩码准备 (ID匹配) ---
    # mask1_flat: [NHW]
    mask1_flat = mask1_HW.view(NHW)
    # mask1_rep: [NHW, 1] (每个源像素的ID)
    mask1_rep = mask1_flat.view(NHW, 1) 
    
    # mask2_rep: [NHW, hw] (广播hw个目标像素的ID，匹配NHW个源像素)
    mask2_rep = mask2_hw.view(N, hw).repeat_interleave(HW, dim=0)

    # valid_j_mask: [NHW, hw]
    # 比较 k-th 源像素ID 和 所有 hw 个目标像素ID
    valid_j_mask = (mask2_rep == mask1_rep) 

    # --- 5. 初始化输出 ---
    corr_sparse = torch.full((NHW, 1, m), torch.nan, device=device, dtype=corr.dtype)
    
    # corr_squeezed: [NHW, hw]
    corr_squeezed = corr.squeeze(1) 

    # --- 6. 邻域采样所需的网格 (在 (h, w) 空间中) ---
    j_h_coords, j_w_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.long), 
        torch.arange(w, device=device, dtype=torch.long), 
        indexing='ij'
    )
    # j_h_coords / j_w_coords 形状 [h, w]

    # --- 7. 循环采样 ---
    for k in range(NHW):
        # j_local_indices: 目标空间 (hw) 中的 1D 索引
        j_local_indices = torch.where(valid_j_mask[k])[0]
        num_matches = j_local_indices.numel()
        
        sampled_corrs = None
        candidate_corrs = None
        
        if num_matches > 0:
            # --- 7.1. ID 匹配成功 ---
            candidate_corrs = corr_squeezed[k, j_local_indices]
            
            if num_matches < m:
                # 重复采样
                rand_indices = torch.randint(0, num_matches, (m,), device=device)
            else:
                # 不重复采样
                rand_indices = torch.randperm(num_matches, device=device)[:m]
            
            sampled_corrs = candidate_corrs[rand_indices]

        else:
            # --- 7.2. ID 匹配失败，执行邻域回退 ---
            
            # 7.2.1. 获取 k 的 (n, h_k, w_k) 坐标 (在 (H, W) 空间)
            n_k = k // HW
            k_local = k % HW
            h_k = k_local // W
            w_k = k_local % W
            
            # 7.2.2. 获取 (H, W) 空间的正向光流 (dx, dy)
            dx = flow1_forward_HW[n_k, 0, h_k, w_k]
            dy = flow1_forward_HW[n_k, 1, h_k, w_k]
            
            # 7.2.3. 计算 (H, W) 空间中的目标坐标 (h_target_HW, w_target_HW)
            w_target_HW = w_k + dx
            h_target_HW = h_k + dy
            
            # 7.2.4. 缩放目标坐标到 (h, w) 空间 (h_t, w_t)
            w_t = w_target_HW * scale_w
            h_t = h_target_HW * scale_h
            
            # 7.2.5. 定义邻域边界并 clamping (在 (h, w) 空间)
            h_min = torch.clamp(torch.round(h_t - r).long(), 0, h - 1)
            h_max = torch.clamp(torch.round(h_t + r).long(), 0, h - 1)
            w_min = torch.clamp(torch.round(w_t - r).long(), 0, w - 1)
            w_max = torch.clamp(torch.round(w_t + r).long(), 0, w - 1)

            # 7.2.6. 从预先计算的 (h, w) 网格中选择邻域内的索引
            if h_min <= h_max and w_min <= w_max:
                # neighborhood_bool_mask: [h, w]
                neighborhood_bool_mask = (j_h_coords >= h_min) & (j_h_coords <= h_max) & \
                                         (j_w_coords >= w_min) & (j_w_coords <= w_max)
                
                # j_local_neighborhood_indices: (hw) 空间中的 1D 索引
                j_local_neighborhood_indices = torch.where(neighborhood_bool_mask.flatten())[0]
                num_neighborhood_matches = j_local_neighborhood_indices.numel()
            else:
                num_neighborhood_matches = 0

            # 7.2.7. 从邻域采样
            if num_neighborhood_matches > 0:
                candidate_corrs = corr_squeezed[k, j_local_neighborhood_indices]
                
                if num_neighborhood_matches < m:
                    rand_indices = torch.randint(0, num_neighborhood_matches, (m,), device=device)
                else:
                    rand_indices = torch.randperm(num_neighborhood_matches, device=device)[:m]
                
                sampled_corrs = candidate_corrs[rand_indices]
                
        # 7.3. 写入结果 (如果为 None，则保持 NaN)
        if sampled_corrs is not None:
            corr_sparse[k, 0, :] = sampled_corrs

    return corr_sparse
def load_ckpt(model, path):
    """ Load checkpoint """
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

def resize_data(img1, img2, flow, factor=1.0):
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode='area')
    img2 = F.interpolate(img2, (h, w), mode='area')
    flow = F.interpolate(flow, (h, w), mode='area') * factor
    return img1, img2, flow

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # grid.shape = [8192, 9, 9, 2]
    
    # 使用PyTorch的grid_sample进行双线性采样
    # img.shape = [B*46*62, 1, 46, 62]
    # grid.shape = [B*46*62, 9, 9, 2]
    img = F.grid_sample(img, grid, align_corners=True)
    # img.shape = [B*46*62, 1, 9, 9]  # 采样后的结果

    # 如果需要掩码，创建有效坐标掩码
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        # mask.shape = [B*46*62, 9, 9, 1]
        return img, mask.float()
        # 返回 img.shape = [B*46*62, 1, 9, 9], mask.shape = [B*46*62, 9, 9, 1]
    
    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1) # [B, 2, H, W]


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def transform(T, p):
    assert T.shape == (4,4)
    return np.einsum('H W j, i j -> H W i', p, T[:3,:3]) + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))

def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0
    
    return flow_01, flow_10

def check_cycle_consistency(flow_01, flow_10):
    flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.1 * min(H, W)).float()
    return mask[0].numpy()