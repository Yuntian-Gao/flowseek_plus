import torch
import torch.nn.functional as F

# 你的原始函数 (V1)
def warp_mask_to_mask2_hw_v1(
    mask1_HW: torch.Tensor,
    flow1_backward_HW: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    V1 (原始方法): 在 (h, w) 网格上采样光流和掩码。
    """
    N, _, H, W = mask1_HW.shape
    device = mask1_HW.device

    h = int(H * scale)
    w = int(W * scale)
    if h < 1 or w < 1:
        raise ValueError(f"Target scale {scale} results in invalid dimensions: ({h}, {w})")

    yy_hw, xx_hw = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij'
    )
    grid_dest_hw = torch.stack((xx_hw, yy_hw), dim=2)
    grid_dest_hw = grid_dest_hw.unsqueeze(0).repeat(N, 1, 1, 1)

    flow1_backward_hw = F.grid_sample(
        flow1_backward_HW,
        grid_dest_hw,
        mode='bilinear', # 光流用双线性插值
        padding_mode='border',
        align_corners=True
    )
    
    flow_perm_hw = flow1_backward_hw.permute(0, 2, 3, 1)

    flow_norm_hw = torch.zeros_like(flow_perm_hw)
    if W > 1:
        flow_norm_hw[..., 0] = flow_perm_hw[..., 0] * (2.0 / (W - 1))
    if H > 1:
        flow_norm_hw[..., 1] = flow_perm_hw[..., 1] * (2.0 / (H - 1))

    grid_src = grid_dest_hw + flow_norm_hw

    mask1_float = mask1_HW.float()
    
    mask2_warped_hw = F.grid_sample(
        mask1_float,
        grid_src,
        mode='nearest', # ID 掩码用 'nearest'
        padding_mode='border',
        align_corners=True
    )
    
    return mask2_warped_hw.long()

def warp_mask_to_mask2_hw_v2(
    mask1_HW: torch.Tensor,
    flow1_backward_HW: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    V2 (新方法): 先在 (H, W) 分辨率上扭曲，然后下采样到 (h, w)。

    Args:
        mask1_HW (torch.Tensor): 形状 [N, 1, H, W]
        flow1_backward_HW (torch.Tensor): 形状 [N, 2, H, W]
        scale (float): 缩放比例
    
    Returns:
        torch.Tensor: 形状 [N, 1, h, w]
    """
    N, _, H, W = mask1_HW.shape
    device = mask1_HW.device

    # 1. 计算目标 (h, w) 维度
    h = int(H * scale)
    w = int(W * scale)
    if h < 1 or w < 1:
        raise ValueError(f"Target scale {scale} results in invalid dimensions: ({h}, {w})")

    # 2. 创建 (H, W) 目标网格
    # 这是用于在 (H, W) 分辨率上进行采样的目标网格
    yy_HW, xx_HW = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    # grid_dest_HW: [N, H, W, 2] (x, y)
    grid_dest_HW = torch.stack((xx_HW, yy_HW), dim=2).unsqueeze(0).repeat(N, 1, 1, 1)

    # 3. 归一化 (H, W) 光流
    flow_perm_HW = flow1_backward_HW.permute(0, 2, 3, 1) # [N, H, W, 2]
    flow_norm_HW = torch.zeros_like(flow_perm_HW)
    if W > 1:
        flow_norm_HW[..., 0] = flow_perm_HW[..., 0] * (2.0 / (W - 1)) # dx -> x_norm
    if H > 1:
        flow_norm_HW[..., 1] = flow_perm_HW[..., 1] * (2.0 / (H - 1)) # dy -> y_norm
    
    # 4. 计算 (H, W) 源网格
    # grid_src_HW: [N, H, W, 2]
    grid_src_HW = grid_dest_HW + flow_norm_HW

    # 5. 在 (H, W) 上采样 (Warp)，生成高分辨率的扭曲掩码
    mask1_float = mask1_HW.float()
    mask2_warped_HW = F.grid_sample(
        mask1_float,
        grid_src_HW,
        mode='nearest', # ID 掩码必须用 'nearest'
        padding_mode='border',
        align_corners=True
    ) # [N, 1, H, W]

    # 6. 将高分辨率的扭曲掩码下采样到 (h, w)
    # 注意: F.interpolate 的 'nearest' 模式与 F.grid_sample 的 'nearest' 
    # 在边界处理和坐标映射上可能存在细微差别。
    mask2_warped_hw = F.interpolate(
        mask2_warped_HW,
        size=(h, w),
        mode='nearest' # 保持 ID
    )
    
    return mask2_warped_hw.long()


# --- 测试 ---
def run_test():
    torch.manual_seed(42)
    
    N, H, W = 1, 10, 10
    scale = 1
    h, w = int(H * scale), int(W * scale)
    
    # 创建一个唯一的 ID 掩码 (0 到 99)
    mask1 = torch.arange(H * W).view(N, 1, H, W).long()
    
    print("--- 测试 1: 零光流 ---")
    flow_zero = torch.zeros(N, 2, H, W).float()
    
    out_v1_zero = warp_mask_to_mask2_hw_v1(mask1, flow_zero, scale)
    out_v2_zero = warp_mask_to_mask2_hw_v2(mask1, flow_zero, scale)
    
    diff_count_zero = torch.sum(out_v1_zero != out_v2_zero).item()
    print(f"V1 (原始) [0,0] 值: {out_v1_zero[0, 0, 0, 0].item()}")
    print(f"V2 (新版) [0,0] 值: {out_v2_zero[0, 0, 0, 0].item()}")
    print(f"零光流: V1 和 V2 之间不同像素的数量: {diff_count_zero} / {N*h*w}")
    # print("V1 (零光流):\n", out_v1_zero.squeeze())
    # print("V2 (零光流):\n", out_v2_zero.squeeze())

    print("\n--- 测试 2: 随机光流 ---")
    # 随机小光流
    flow_random = (torch.rand(N, 2, H, W) - 0.5) * 4.0 
    
    out_v1_rand = warp_mask_to_mask2_hw_v1(mask1, flow_random, scale)
    out_v2_rand = warp_mask_to_mask2_hw_v2(mask1, flow_random, scale)
    
    diff_count_rand = torch.sum(out_v1_rand != out_v2_rand).item()
    print(f"随机光流: V1 和 V2 之间不同像素的数量: {diff_count_rand} / {N*h*w}")
    # print("V1 (随机光流):\n", out_v1_rand.squeeze())
    # print("V2 (随机光流):\n", out_v2_rand.squeeze())

    if diff_count_zero > 0 or diff_count_rand > 0:
        print("\n结论: 两个函数不一致。")
    else:
        print("\n结论: 两个函数在此测试中一致。")

if __name__ == "__main__":
    run_test()