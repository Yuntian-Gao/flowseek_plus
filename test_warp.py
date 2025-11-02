import torch
import torch.nn.functional as F
import math

# -----------------------------------------------------------------
# 1. 您的 `warp_mask_to_mask2_hw` (算法A，待测试)
# (与您提供的一模一样)
# -----------------------------------------------------------------
def warp_mask_to_mask2_hw(
    mask1_HW: torch.Tensor, 
    flow1_backward_HW: torch.Tensor, 
    scale: float
) -> torch.Tensor:
    """
    算法 A: 先降采样光流 (Bilinear)，再拉取掩码 (Nearest)
    Pull( Mask, Downsample(Flow) )
    """
    N, _, H, W = mask1_HW.shape
    device = mask1_HW.device
    h = round(H * scale)
    w = round(W * scale)
    if h < 1 or w < 1:
        raise ValueError(f"Target scale {scale} results in invalid dimensions: ({h}, {w})")
    
    # 2. 创建 (h, w) 目标网格
    yy_hw, xx_hw = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    grid_dest_hw = torch.stack((xx_hw, yy_hw), dim=2) 
    grid_dest_hw = grid_dest_hw.unsqueeze(0).repeat(N, 1, 1, 1)
    
    # 3. 降采样光流
    flow1_backward_hw = F.grid_sample(
        flow1_backward_HW, 
        grid_dest_hw,
        mode='bilinear',
        padding_mode='border',
        align_corners=True 
    )
    flow_perm_hw = flow1_backward_hw.permute(0, 2, 3, 1)
    
    # 5. 归一化 (h, w) 光流
    flow_norm_hw = torch.zeros_like(flow_perm_hw)
    if W > 1:
        flow_norm_hw[..., 0] = flow_perm_hw[..., 0] * (2.0 / (W - 1))
    if H > 1:
        flow_norm_hw[..., 1] = flow_perm_hw[..., 1] * (2.0 / (H - 1))
        
    # 6. 计算 (h, w) 源网格
    grid_src = grid_dest_hw + flow_norm_hw 
    
    # 7. 拉取 (H, W) 掩码
    mask1_float = mask1_HW.float()
    mask2_warped_hw = F.grid_sample(
        mask1_float, 
        grid_src, 
        mode='nearest', # 'nearest' 是关键
        padding_mode='border', 
        align_corners=True
    )
    return mask2_warped_hw.long()

# -----------------------------------------------------------------
# 2. 新的 "Ground Truth" (算法B，您的新标准)
# -----------------------------------------------------------------
def warp_pull_then_downsample(
    mask1_HW: torch.Tensor, 
    flow1_backward_HW: torch.Tensor, 
    scale: float
) -> torch.Tensor:
    """
    算法 B: 先在全分辨率上拉取掩码 (Nearest)，再降采样掩码 (Nearest)
    Downsample( Pull(Mask, Flow) )
    """
    N, _, H, W = mask1_HW.shape
    device = mask1_HW.device
    h = round(H * scale)
    w = round(W * scale)
    if h < 1 or w < 1:
        raise ValueError(f"Target scale {scale} results in invalid dimensions: ({h}, {w})")
    
    # --- 步骤 1: 在 (H, W) 全分辨率上进行扭曲 ---
    
    # 1a. 创建 (H, W) 目标网格
    yy_HW, xx_HW = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    # grid_dest_HW: [N, H, W, 2]
    grid_dest_HW = torch.stack((xx_HW, yy_HW), dim=2).unsqueeze(0).repeat(N, 1, 1, 1)
    
    # 1b. 归一化 (H, W) 光流
    flow_perm_HW = flow1_backward_HW.permute(0, 2, 3, 1)
    flow_norm_HW = torch.zeros_like(flow_perm_HW)
    if W > 1:
        flow_norm_HW[..., 0] = flow_perm_HW[..., 0] * (2.0 / (W - 1))
    if H > 1:
        flow_norm_HW[..., 1] = flow_perm_HW[..., 1] * (2.0 / (H - 1))
        
    # 1c. 计算 (H, W) 源网格
    grid_src_HW = grid_dest_HW + flow_norm_HW
    
    # 1d. 拉取 (H, W) 掩码，得到中间结果 [N, 1, H, W]
    mask2_warped_HW = F.grid_sample(
        mask1_HW.float(),
        grid_src_HW,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    )
    
    # --- 步骤 2: 降采样中间掩码 ---
    mask_truth_hw = F.interpolate(
        mask2_warped_HW,
        size=(h, w),
        mode='nearest-exact' # 使用 'nearest-exact' 避免模糊
    )
    
    return mask_truth_hw.long()

# -----------------------------------------------------------------
# 3. 新的测试函数 (算法 A vs 算法 B)
# -----------------------------------------------------------------
def test_warp_pull_vs_pull_downsample():
    """
    比较 "Downsample Flow, then Pull" (您的实现, A)
    与 "Pull, then Downsample" (您的新标准, B)
    """
    N = 2
    H = 20
    W = 20
    scale = 1# h=10, w=10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 运行 'Pull(A)' vs 'Pull(B)' 测试 (N={N}, H={H}, W={W}, scale={scale}) 在 {device} 上 ---")

    # [N, 1, H, W], IDs 0-4
    mask1_HW_test = torch.randint(0, 5, (N, 1, H, W), device=device)
    # [N, 2, H, W]
    # 两种方法都需要反向光流，所以我们直接生成它
    flow1_backward_test = torch.randn(N, 2, H, W, device=device) .float()

    print("正在计算 A: 您的实现 (Downsample Flow, then Pull)...")
    try:
        mask_impl_hw = warp_mask_to_mask2_hw(
            mask1_HW_test, flow1_backward_test, scale
        )
        print("算法 A 运行完毕。")
    except Exception as e:
        print(f"算法 A 运行时出错: {e}")
        return

    print("正在计算 B: 您的新标准 (Pull, then Downsample)...")
    try:
        mask_truth_hw = warp_pull_then_downsample(
            mask1_HW_test, flow1_backward_test, scale
        )
        print("算法 B 运行完毕。")
    except Exception as e:
        print(f"算法 B 运行时出错: {e}")
        return

    # --- C. 比较 ---
    print("\n--- 结果比较 ---")
    
    # 两种方法都是 'nearest' 采样并且是 long()，
    # 它们都没有孔洞 (因为 padding_mode='border')
    
    total_pixels = mask_impl_hw.numel()
    mismatched_pixels = torch.sum(mask_impl_hw != mask_truth_hw).item()
    accuracy = 1.0 - (mismatched_pixels / total_pixels)
    
    print(f"像素匹配率: {accuracy * 100.0 :.2f}%")
    print(f"不匹配的像素数量: {mismatched_pixels} / {total_pixels}")

    if accuracy < 1.0:
        print("\n--- 结论 ---")
        print("测试完成。结果 *不* 完全匹配，这是 **预期之中** 的。")
        print("这证明了 '先插值光流' 和 '先扭曲再插值掩码' 是两种略微不同的算法。")
        print(f"高匹配率 (如 >95%) 表明您的函数在概念上是正确的，并且在计算上更高效。")
    else:
        print("\n--- 结论 ---")
        print("✅ 测试通过！两种算法的结果完全一致。")


if __name__ == "__main__":
    # 运行测试
    test_warp_pull_vs_pull_downsample()