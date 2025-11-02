import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 您的原始函数 (待测试)
# ---------------------------------------------------------------------------

def sample_correlation_from_masks_with_neighborhood_fallback_vectorized(
    corr: torch.Tensor, 
    mask1_HW: torch.Tensor, 
    mask2_hw: torch.Tensor, 
    flow1_forward_HW: torch.Tensor, 
    m: int,
    r: float
) -> torch.Tensor:
    """
    根据 mask1 (高分辨率) 和 mask2 (低分辨率) 采样代价体 corr。(向量化版本)
    
    如果ID匹配失败（匹配数 < m），则在 flow1_forward (高分辨率) 预测的、并缩放至
    低分辨率空间的邻域 (半径 r) 内进行 *规则网格* 采样。
    (此版本要求 m 必须等于 (2*r+1)^2)

    Args:
        corr (torch.Tensor): 代价体, 形状 [NHW, 1, hw]
        mask1_HW (torch.Tensor): 图像1的掩码 (高分辨率), 形状 [N, 1, H, W]
        mask2_hw (torch.Tensor): 图像2的掩码 (低分辨率), 形状 [N, 1, h, w]
        flow1_forward_HW (torch.Tensor): *正向* 光流 (1 -> 2), 
                                      形状 [N, 2, H, W],
                                      光流值(dx, dy)定义在 (H, W) 坐标系中。
                                      通道 0: dx, 通道 1: dy
        m (int): 每个点要采样的数量。必须满足 m = (2*r+1)^2
        r (float): 邻域回退采样的半径 (在 (h, w) 空间中定义)。
                   (2*r+1) 必须是一个整数。

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
    dtype = corr.dtype

    # --- 2. 形状验证 ---
    corr = corr.view(NHW, 1, hw) 
    
    # corr_squeezed: [NHW, hw]
    corr_squeezed = corr.squeeze(1) 

    # --- 3. 尺度因子 ---
    scale_h = h / H
    scale_w = w / W

    # --- 4. 掩码准备 (ID匹配) ---
    
    # mask1_flat: [NHW]
    mask1_flat = mask1_HW.view(NHW)
    # mask1_rep: [NHW, 1] (每个源像素的ID)
    mask1_rep = mask1_flat.view(NHW, 1) 
    
    # mask2_rep: [NHW, hw] (广播hw个目标像素的ID，匹配NHW个源像素)
    mask2_rep = mask2_hw.view(N, hw).repeat_interleave(HW, dim=0)

    # valid_j_mask: [NHW, hw] (bool)
    # 比较 k-th 源像素ID 和 所有 hw 个目标像素ID
    valid_j_mask = (mask2_rep == mask1_rep) 
    
    # --- 5. 找出需要回退的像素 (Fallback) vs ID匹配的像素 (ID-Match) ---
    
    # num_matches_per_k: [NHW]
    num_matches_per_k = valid_j_mask.sum(dim=1)
    
    # fallback_mask_k: [NHW] (bool)
    fallback_mask_k = (num_matches_per_k < m)
    
    # id_match_mask_k: [NHW] (bool)
    id_match_mask_k = ~fallback_mask_k

    # fallback_indices_k: [K2] (K2是需要回退的像素总数)
    fallback_indices_k = torch.where(fallback_mask_k)[0]
    
    # id_match_indices_k: [K1] (K1是ID匹配的像素总数)
    id_match_indices_k = torch.where(id_match_mask_k)[0]

    K1 = id_match_indices_k.numel()
    K2 = fallback_indices_k.numel()

    # --- 6. 初始化输出 ---
    corr_sparse = torch.full((NHW, 1, m), torch.nan, device=device, dtype=dtype)

    # =========================================================================
    # 7. 批处理：ID 匹配 (K1 个点)
    # =========================================================================
    if K1 > 0:
        # 7.1. 只选择需要ID匹配的行
        # valid_j_mask_k1: [K1, hw]
        valid_j_mask_k1 = valid_j_mask[id_match_indices_k]
        
        # corr_k1: [K1, hw]
        corr_k1 = corr_squeezed[id_match_indices_k]

        # 7.2. 使用 "topk + 随机数" 技巧进行批处理采样
        
        # rand_vals: [K1, hw] (在 [0, 1] 之间)
        # !! 这是唯一的随机性来源 !!
        rand_vals = torch.rand(K1, hw, device=device, dtype=dtype)
        
        # 只保留 ID 匹配的位置的随机数，其他设为 -1
        rand_vals.masked_fill_(~valid_j_mask_k1, -1.0)
        
        # 7.3. 选取 top m 个随机数对应的索引
        # _, sampled_j_indices_k1: [K1, m]
        _, sampled_j_indices_k1 = torch.topk(rand_vals, k=m, dim=1)
        
        # 7.4. 使用 gather 从代价体中提取采样的值
        # sampled_corrs_k1: [K1, m]
        sampled_corrs_k1 = torch.gather(corr_k1, 1, sampled_j_indices_k1)
        
        # 7.5. 填入
        corr_sparse[id_match_indices_k] = sampled_corrs_k1.unsqueeze(1)


    # =========================================================================
    # 8. 批处理：邻域回退 (K2 个点)
    # =========================================================================
    if K2 > 0:
        # 8.1. 获取 K2 个点的 (n, h_k, w_k) 坐标 (在 (H, W) 空间)
        n_k2 = fallback_indices_k // HW
        k_local_k2 = fallback_indices_k % HW
        h_k2 = k_local_k2 // W
        w_k2 = k_local_k2 % W
        
        # 8.2. 获取 (H, W) 空间的正向光流 (dx, dy)
        # flow1_forward_HW: [N, 2, H, W]
        # 使用高级索引获取 [K2, 2]
        flow_k2 = flow1_forward_HW[n_k2, :, h_k2, w_k2]
        dx_k2 = flow_k2[:, 0] # [K2]
        dy_k2 = flow_k2[:, 1] # [K2]

        # 8.3. 计算 (H, W) 空间中的目标坐标 (w_target_HW, h_target_HW)
        # w_k2 和 h_k2 需要转为 float
        w_target_HW_k2 = w_k2.to(dtype) + dx_k2
        h_target_HW_k2 = h_k2.to(dtype) + dy_k2
        
        # 8.4. 缩放目标坐标到 (h, w) 空间 (h_t, w_t)
        # w_t_k2, h_t_k2: [K2]
        w_t_k2 = w_target_HW_k2 * scale_w
        h_t_k2 = h_target_HW_k2 * scale_h

        # 8.5. 生成 m 个邻域采样点 (使用 grid_sample)
        
        # ==================== (!! 已修改 !!) ====================
        # 8.5.1. 生成 (2r+1)x(2r+1) 的规则网格偏移量
        side_length_float = 2 * r + 1
        side_length = int(round(side_length_float))
        
        # 验证 m 和 r 是否匹配
        if side_length**2 != m:
            raise ValueError(
                f"Fallback sampling m ({m}) must equal (2*r+1)^2 "
                f"({side_length**2} for r={r})."
            )
       
            
        # E.g., r=1, side_length=3, offsets_1d = [-1, 0, 1]
        # E.g., r=2, side_length=5, offsets_1d = [-2, -1, 0, 1, 2]
        offsets_1d = torch.linspace(-r, r, steps=side_length, device=device, dtype=dtype)
        
        # grid_h_offsets, grid_w_offsets: [side_length, side_length]
        grid_h_offsets, grid_w_offsets = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
        
        # offsets_h_m, offsets_w_m: [m]
        offsets_h_m = grid_h_offsets.reshape(m)
        offsets_w_m = grid_w_offsets.reshape(m)

        # 8.5.2. 计算采样坐标 (w_s, h_s)
        # w_t_k2: [K2] -> [K2, 1]
        # offsets_w_m: [m] -> [1, m]
        # w_s_k2m, h_s_k2m: [K2, m] (应用广播)
        w_s_k2m = w_t_k2.view(K2, 1) + offsets_w_m.view(1, m)
        h_s_k2m = h_t_k2.view(K2, 1) + offsets_h_m.view(1, m)
        # ================== (!! 修改结束 !!) ==================

        # 8.5.3. 归一化坐标以用于 grid_sample (从 [0, W-1] 映射到 [-1, 1])
        # grid_sample 需要 (x, y) 顺序，即 (w, h)
        grid_w = (w_s_k2m / (w - 1)) * 2 - 1
        grid_h = (h_s_k2m / (h - 1)) * 2 - 1
        
        # grid: [K2, 1, m, 2] (grid_sample 需要 N, H_out, W_out, 2)
        # 这里我们的 N=K2, H_out=1, W_out=m
        grid = torch.stack((grid_w, grid_h), dim=-1).view(K2, 1, m, 2)
        
        # 8.5.4. 准备 grid_sample 的输入 (代价体)
        # corr_squeezed: [NHW, hw]
        # corr_k2: [K2, hw]
        corr_k2 = corr_squeezed[fallback_indices_k]
        
        # corr_k2_hw_view: [K2, 1, h, w] (N, C, H_in, W_in)
        corr_k2_hw_view = corr_k2.view(K2, 1, h, w)
        
        # 8.5.5. 执行采样
        # sampled_corrs_k2: [K2, 1, 1, m]
        sampled_corrs_k2 = F.grid_sample(
            corr_k2_hw_view, 
            grid, 
            mode='bilinear', # 使用双线性插值
            padding_mode='border', # 'border' 会 clamp 到边界
            align_corners=True # 坐标 [0, W-1] 对应 [-1, 1]
        )
        
        # 8.5.6. 整理形状并填入
        # .view(K2, 1, m)
        corr_sparse[fallback_indices_k] = sampled_corrs_k2.view(K2, 1, m)

    # -------------------------------------------------------------------------
    
    return corr_sparse


# ---------------------------------------------------------------------------
# 暴力循环参考实现 (用于测试)
# ---------------------------------------------------------------------------

def sample_correlation_reference_loop(
    corr: torch.Tensor, 
    mask1_HW: torch.Tensor, 
    mask2_hw: torch.Tensor, 
    flow1_forward_HW: torch.Tensor, 
    m: int,
    r: float
) -> torch.Tensor:
    """
    与向量化版本逻辑完全相同的 "暴力" 循环参考实现。
    """
    
    # --- 1. 形状提取 ---
    N, _, H, W = mask1_HW.shape
    _, _, h, w = mask2_hw.shape
    
    HW = H * W
    hw = h * w
    NHW = N * H * W
    device = corr.device
    dtype = corr.dtype

    # --- 2. 形状验证 ---
    corr_squeezed = corr.view(NHW, hw) 

    # --- 3. 尺度因子 ---
    scale_h = h / H
    scale_w = w / W

    # --- 4. & 5. 预计算所有 K1 和 K2 索引 ---
    # (这必须与向量化版本完全一致地完成)
    mask1_flat = mask1_HW.view(NHW)
    mask1_rep = mask1_flat.view(NHW, 1) 
    mask2_rep = mask2_hw.view(N, hw).repeat_interleave(HW, dim=0)
    valid_j_mask = (mask2_rep == mask1_rep) 
    num_matches_per_k = valid_j_mask.sum(dim=1)
    fallback_mask_k = (num_matches_per_k < m)
    id_match_mask_k = ~fallback_mask_k
    
    id_match_indices_k = torch.where(id_match_mask_k)[0]
    K1 = id_match_indices_k.numel()

    # 创建一个从 k (全局索引) 到 k1 (K1批次中的索引) 的映射
    # e.g., id_match_indices_k = [5, 8, 12]
    # k=5 -> k1_idx=0
    # k=8 -> k1_idx=1
    # k=12 -> k1_idx=2
    k_to_k1_map = {k.item(): i for i, k in enumerate(id_match_indices_k)}

    # --- 6. 初始化输出 ---
    corr_sparse_ref = torch.full((NHW, 1, m), torch.nan, device=device, dtype=dtype)

    # =========================================================================
    # 7. 预生成 *单个* 随机张量 (!! 关键 !!)
    # =========================================================================
    rand_vals_k1 = torch.rand(K1, hw, device=device, dtype=dtype)

    # =========================================================================
    # 8. 预生成回退 (Fallback) 邻域偏移量
    # =========================================================================
    side_length = int(round(2 * r + 1))
    if side_length**2 != m:
        raise ValueError("m 和 r 不匹配")
        
    offsets_1d = torch.linspace(-r, r, steps=side_length, device=device, dtype=dtype)
    grid_h_offsets, grid_w_offsets = torch.meshgrid(offsets_1d, offsets_1d, indexing='ij')
    offsets_h_m = grid_h_offsets.reshape(m)
    offsets_w_m = grid_w_offsets.reshape(m)
    
    # =========================================================================
    # 9. 开始逐像素 (k) 循环
    # =========================================================================
    
    for k in range(NHW):
        n = k // HW
        k_local = k % HW
        h_k = k_local // W
        w_k = k_local % W
        
        # --- 检查 k 属于 K1 (ID-Match) 还是 K2 (Fallback) ---
        
        if k in k_to_k1_map:
            # --- 9.A. ID 匹配路径 ---
            
            # 7.1. 获取行
            k1_idx = k_to_k1_map[k]
            corr_k = corr_squeezed[k] # [hw]
            valid_j_mask_k = valid_j_mask[k] # [hw]
            
            # 7.2. 获取预先生成的随机数
            # (我们必须复制 rand_vals，因为 masked_fill_ 是 in-place)
            rand_vals_k = rand_vals_k1[k1_idx].clone() # [hw]
            rand_vals_k.masked_fill_(~valid_j_mask_k, -1.0)

            # 7.3. Top-k
            _, sampled_j_indices_k = torch.topk(rand_vals_k, k=m, dim=0) # [m]
            
            # 7.4. Gather
            sampled_corrs_k = corr_k[sampled_j_indices_k] # [m]
            
            # 7.5. 填入
            corr_sparse_ref[k, 0, :] = sampled_corrs_k

        else:
            # --- 9.B. 邻域回退路径 (确定性) ---
            
            # 8.2. 获取光流
            flow_k = flow1_forward_HW[n, :, h_k, w_k] # [2]
            dx_k = flow_k[0].item()
            dy_k = flow_k[1].item()
            
            # 8.3. (H, W) 目标坐标
            w_target_HW_k = float(w_k) + dx_k
            h_target_HW_k = float(h_k) + dy_k
            
            # 8.4. (h, w) 目标坐标
            w_t_k = w_target_HW_k * scale_w
            h_t_k = h_target_HW_k * scale_h
            
            # 8.5.2. 计算采样坐标 (w_s, h_s)
            # w_s_k_m, h_s_k_m: [m]
            w_s_k_m = w_t_k + offsets_w_m
            h_s_k_m = h_t_k + offsets_h_m
            
            # 8.5.3. 归一化
            grid_w_k = (w_s_k_m / (w - 1)) * 2 - 1
            grid_h_k = (h_s_k_m / (h - 1)) * 2 - 1
            
            # grid_k: [1, 1, m, 2] (N=1, H_out=1, W_out=m)
            grid_k = torch.stack((grid_w_k, grid_h_k), dim=-1).view(1, 1, m, 2)
            
            # 8.5.4. 准备输入
            corr_k = corr_squeezed[k] # [hw]
            corr_k_hw_view = corr_k.view(1, 1, h, w) # [1, 1, h, w]
            
            # 8.5.5. 采样
            sampled_corrs_k = F.grid_sample(
                corr_k_hw_view,
                grid_k,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ) # [1, 1, 1, m]
            
            # 8.5.6. 填入
            corr_sparse_ref[k, 0, :] = sampled_corrs_k.view(m)

    return corr_sparse_ref

# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def run_test():
    
    # --- 1. 参数设置 ---
    _SEED = 42
    N, H, W = 2, 10, 10
    scale = 0.5
    h, w = int(H * scale), int(W * scale)
    
    r = 2
    m = int((2 * r + 1)**2) # m = 9
    
    HW = H * W
    hw = h * w
    NHW = N * HW
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # F.grid_sample 在 float64 上的实现可能因平台而异，统一用 float32
    dtype = torch.float32 
    torch.set_default_dtype(dtype)
    
    print(f"--- 测试开始 ---")
    print(f"设备: {device},N={N}, H={H}, W={W}, h={h}, w={w}, m={m}, r={r}")

    # --- 2. 创建测试数据 ---
    
    # 我们将创建一种“混合”场景：
    # Batch 0: mask1 [0..99], mask2 [0..24]
    #   -> 约 1/4 的 k 会 ID-Match, 3/4 会 Fallback
    # Batch 1: mask1 [100..199], mask2 [500..524]
    #   -> 0 个 k 会 ID-Match, 全部会 Fallback
    
    torch.manual_seed(_SEED)
    
    # Mask 1 (高分辨率)
    mask1_n0 = torch.arange(HW, device=device).view(1, 1, H, W)
    mask1_n1 = torch.arange(HW, device=device).view(1, 1, H, W) + 100
    mask1_HW = torch.cat([mask1_n0, mask1_n1], dim=0).long()
    
    # Mask 2 (低分辨率)
    mask2_n0 = torch.arange(hw, device=device).view(1, 1, h, w)
    mask2_n1 = torch.arange(hw, device=device).view(1, 1, h, w) + 500
    mask2_hw = torch.cat([mask2_n0, mask2_n1], dim=0).long()
    
    # 正向光流
    flow1_forward_HW = (torch.rand(N, 2, H, W, device=device) - 0.5) * 4.0
    
    # 代价体
    corr = torch.randn(NHW, 1, hw, device=device)
    
    # --- 3. 运行向量化版本 ---
    # 我们必须重置种子，因为数据生成消耗了 RNG
    torch.manual_seed(_SEED) 
    out_vec = sample_correlation_from_masks_with_neighborhood_fallback_vectorized(
        corr, mask1_HW, mask2_hw, flow1_forward_HW, m, r
    )
    
    # --- 4. 运行暴力循环版本 ---
    # 再次重置种子，以确保 K1 批次的 torch.rand() 完全相同
    torch.manual_seed(_SEED) 
    out_ref = sample_correlation_reference_loop(
        corr, mask1_HW, mask2_hw, flow1_forward_HW, m, r
    )
    
    # --- 5. 比较 ---
    
    # 由于两个版本都使用 torch.full(..., torch.nan, ...) 初始化，
    # 并且都应该填充所有条目，因此 nan 的位置也应该完全匹配。
    is_close = torch.allclose(out_vec, out_ref, equal_nan=True)
    
    print(f"\n--- 测试结果 ---")
    print(f"向量化版本 (out_vec) Shape: {out_vec.shape}")
    print(f"参考版本 (out_ref) Shape: {out_ref.shape}")
    print(f"所有值是否严格接近 (equal_nan=True): {is_close}")

    if not is_close:
        print("\n!! 失败 !! 两个版本的输出不一致。")
        # 找出第一个不匹配的 k
        diff = ~torch.isclose(out_vec, out_ref, equal_nan=True)
        first_diff_k = torch.where(diff.any(dim=-1).any(dim=-1))[0][0].item()
        
        print(f"第一个不匹配的索引 k = {first_diff_k}")
        print(f"Vec[{first_diff_k}]:\n{out_vec[first_diff_k].squeeze()}")
        print(f"Ref[{first_diff_k}]:\n{out_ref[first_diff_k].squeeze()}")
    else:
        print("\n>> 成功! 向量化版本和循环参考版本完全一致。")

if __name__ == "__main__":
    run_test()