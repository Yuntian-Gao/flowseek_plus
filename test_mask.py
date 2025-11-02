import torch
import math

# -----------------------------------------------------------------
# 1. 您的原始函数 (待测试)
# -----------------------------------------------------------------
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
    5. 最终的反向光流 = flow_backward_sum / (weights_sum + epsilon)。
    
    (修改版): 孔洞 (weights_sum == 0) 将被填充为 -1。
    """
    N, _, H, W = flow_forward.shape
    device = flow_forward.device
    dtype = flow_forward.dtype

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
    
    # [N, 2, H, W]
    v_flow = torch.stack([v_dx, v_dy], dim=1)

    # 4. 初始化输出累加器
    flow_backward_sum = torch.zeros(N, 2, H, W, device=device, dtype=dtype)
    weights_sum = torch.zeros(N, 1, H, W, device=device, dtype=dtype)

    # 5. 计算 p2 的四个最近邻整数坐标
    x2_floor = torch.floor(x2).long()
    y2_floor = torch.floor(y2).long()
    x2_ceil = x2_floor + 1
    y2_ceil = y2_floor + 1

    # 6. 计算双线性散布权重
    w_tl = (x2_ceil.float() - x2) * (y2_ceil.float() - y2) # top-left
    w_tr = (x2 - x2_floor.float()) * (y2_ceil.float() - y2) # top-right
    w_bl = (x2_ceil.float() - x2) * (y2 - y2_floor.float()) # bottom-left
    w_br = (x2 - x2_floor.float()) * (y2 - y2_floor.float()) # bottom-right

    # 7. 定义一个辅助函数来执行 scatter_add_
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
        flat_idx_flow = flat_idx.unsqueeze(1).expand(-1, 2, -1, -1).view(N, 2, -1)
        
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

    # -----------------------------------------------------------------
    # 9. 归一化，并将孔洞填充为 -1
    # -----------------------------------------------------------------
    
    epsilon = 1e-8
    
    # (A) 识别孔洞
    hole_mask = (weights_sum < epsilon)

    # (B) 归一化
    flow_backward_normalized = flow_backward_sum / (weights_sum + epsilon)

    # (C) 填充孔洞
    flow_backward = torch.where(
        hole_mask, 
        -1.0, 
        flow_backward_normalized
    )
    # -----------------------------------------------------------------

    return flow_backward

# -----------------------------------------------------------------
# 2. "Ground Truth" 函数 (使用循环，易于理解)
# -----------------------------------------------------------------
def forward_flow_to_backward_ground_truth(flow_forward: torch.Tensor) -> torch.Tensor:
    """
    使用简单的 Python 循环实现与 `forward_flow_to_backward` 完全相同的逻辑。
    这会非常慢，但易于验证其正确性。
    """
    N, _, H, W = flow_forward.shape
    device = flow_forward.device
    dtype = flow_forward.dtype
    
    # 4. 初始化输出累加器
    flow_backward_sum = torch.zeros(N, 2, H, W, device=device, dtype=dtype)
    weights_sum = torch.zeros(N, 1, H, W, device=device, dtype=dtype)
    
    epsilon = 1e-8
    
    # 遍历所有批次
    for n in range(N):
        # 遍历帧1中的每一个源像素 (x1, y1)
        for y1 in range(H):
            for x1 in range(W):
                
                # 2. 计算 p2 目标坐标 (x2, y2)
                dx = flow_forward[n, 0, y1, x1].item()
                dy = flow_forward[n, 1, y1, x1].item()
                x2 = float(x1) + dx
                y2 = float(y1) + dy
                
                # 3. 获取要散布的值 V = -flow_fwd(p1)
                v_dx = -dx
                v_dy = -dy
                
                # 5. 计算 p2 的四个最近邻整数坐标
                x2_floor = math.floor(x2)
                y2_floor = math.floor(y2)
                x2_ceil = x2_floor + 1
                y2_ceil = y2_floor + 1

                # 6. 计算双线性散布权重
                w_tl = (float(x2_ceil) - x2) * (float(y2_ceil) - y2)
                w_tr = (x2 - float(x2_floor)) * (float(y2_ceil) - y2)
                w_bl = (float(x2_ceil) - x2) * (y2 - float(y2_floor))
                w_br = (x2 - float(x2_floor)) * (y2 - float(y2_floor))

                # 8. 向四个角进行散布 (带边界检查)
                
                # Top-Left
                if 0 <= x2_floor < W and 0 <= y2_floor < H:
                    weights_sum[n, 0, y2_floor, x2_floor] += w_tl
                    flow_backward_sum[n, 0, y2_floor, x2_floor] += v_dx * w_tl
                    flow_backward_sum[n, 1, y2_floor, x2_floor] += v_dy * w_tl

                # Top-Right
                if 0 <= x2_ceil < W and 0 <= y2_floor < H:
                    weights_sum[n, 0, y2_floor, x2_ceil] += w_tr
                    flow_backward_sum[n, 0, y2_floor, x2_ceil] += v_dx * w_tr
                    flow_backward_sum[n, 1, y2_floor, x2_ceil] += v_dy * w_tr

                # Bottom-Left
                if 0 <= x2_floor < W and 0 <= y2_ceil < H:
                    weights_sum[n, 0, y2_ceil, x2_floor] += w_bl
                    flow_backward_sum[n, 0, y2_ceil, x2_floor] += v_dx * w_bl
                    flow_backward_sum[n, 1, y2_ceil, x2_floor] += v_dy * w_bl
                
                # Bottom-Right
                if 0 <= x2_ceil < W and 0 <= y2_ceil < H:
                    weights_sum[n, 0, y2_ceil, x2_ceil] += w_br
                    flow_backward_sum[n, 0, y2_ceil, x2_ceil] += v_dx * w_br
                    flow_backward_sum[n, 1, y2_ceil, x2_ceil] += v_dy * w_br

    # -----------------------------------------------------------------
    # 9. 归一化，并将孔洞填充为 -1 (与原始函数完全相同的逻辑)
    # -----------------------------------------------------------------
    
    # (A) 识别孔洞
    hole_mask = (weights_sum < epsilon)

    # (B) 归一化
    flow_backward_normalized = flow_backward_sum / (weights_sum + epsilon)

    # (C) 填充孔洞
    flow_backward_truth = torch.where(
        hole_mask, 
        -1.0, 
        flow_backward_normalized
    )
    
    return flow_backward_truth


# -----------------------------------------------------------------
# 3. 测试运行函数
# -----------------------------------------------------------------
def test_forward_to_backward():
    """
    运行测试，比较两个函数的输出。
    """
    # --- 1. 设置测试参数 ---
    # 使用较小的 H 和 W，因为 ground truth 函数非常慢
    N = 2
    H = 16
    W = 16
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 运行测试 (N={N}, H={H}, W={W}) 在 {device} 上 ---")
    
    # 创建随机前向光流
    # (乘以 H/4 使光流范围更大，更有可能产生遮挡和OOB)
    flow_forward_test = (torch.randn(N, 2, H, W, device=device) * (H / 4)).float()
    
    # (可选) 创建一个已知会导致孔洞的特定测试用例
    # flow_forward_test = torch.zeros(N, 2, H, W, device=device)
    # flow_forward_test[0, :, 0, 0] = 1000.0 # 将一个像素移出边界，产生孔洞
    
    # --- 2. 运行两个函数 ---
    print("正在运行您的 (scatter_add) 实现...")
    try:
        flow_backward_impl = forward_flow_to_backward(flow_forward_test)
        print("您的实现运行完毕。")
    except Exception as e:
        print(f"您的实现运行时出错: {e}")
        return

    print("正在运行 'Ground Truth' (循环) 实现 (可能需要几秒钟)...")
    try:
        flow_backward_truth = forward_flow_to_backward_ground_truth(flow_forward_test)
        print("'Ground Truth' 实现运行完毕。")
    except Exception as e:
        print(f"'Ground Truth' 实现运行时出错: {e}")
        return
        
    # --- 3. 比较结果 ---
    print("\n--- 结果比较 ---")
    
    # 检查1: 比较孔洞 (-1.0) 的位置
    # 我们直接比较等于-1.0，因为这是由 torch.where 精确插入的
    hole_mask_impl = (flow_backward_impl == -1.0)
    hole_mask_truth = (flow_backward_truth == -1.0)
    
    holes_match = torch.all(hole_mask_impl == hole_mask_truth)
    print(f"孔洞位置是否完全匹配: {holes_match}")
    if not holes_match:
        print("!! 失败: 孔洞位置不匹配。")
        print(f"   您的实现找到 {hole_mask_impl.sum()} 个孔洞。")
        print(f"   Ground Truth 找到 {hole_mask_truth.sum()} 个孔洞。")
        # (可选：打印出不匹配的索引)
        # print(torch.nonzero(hole_mask_impl != hole_mask_truth))
        return

    # 检查2: 比较非孔洞区域的数值
    non_hole_mask = ~hole_mask_truth
    
    # 提取所有非孔洞的有效值
    # (我们只提取通道0，因为如果通道0是孔洞，通道1也必须是)
    valid_impl_values = flow_backward_impl[non_hole_mask]
    valid_truth_values = flow_backward_truth[non_hole_mask]

    if valid_truth_values.numel() > 0:
        # 使用 allclose 进行数值比较 (允许微小的浮点误差)
        # atol (absolute tolerance) 设为 1e-6 比较稳妥
        are_close = torch.allclose(valid_impl_values, valid_truth_values, atol=1e-6)
        print(f"非孔洞数值是否接近 (atol=1e-6): {are_close}")
        
        # 计算平均绝对误差 (MAE)
        mae = torch.abs(valid_impl_values - valid_truth_values).mean().item()
        print(f"非孔洞数值的平均绝对误差 (MAE): {mae: .2e}")
        
        if not are_close:
            print("!! 失败: 非孔洞数值存在显著差异。")
            max_diff = torch.abs(valid_impl_values - valid_truth_values).max().item()
            print(f"   最大差异: {max_diff: .2e}")
            return
    else:
        print("图像中没有非孔洞区域（可能所有像素都移出边界）。")

    print("\n--- ✅ 测试通过 ---")
    print("您的 `scatter_add_` 实现与 'Ground Truth' 循环实现的结果一致。")


if __name__ == "__main__":
    # 运行测试
    test_forward_to_backward()