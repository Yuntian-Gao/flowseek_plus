#!/bin/bash

# 默认值

GPUS="5"
FS_PP=1
# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --size)
      SIZE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --fsPP)
      FS_PP="$2"
      shift 2
      ;;
  esac
done

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES=$GPUS

# echo "Validation FlowSeek_chairs (T)"
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_C_1.pth_20251101124756 --dataset sintel --fsPP $FS_PP
python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_C_1.pth_20251101124756 --dataset kitti  --fsPP $FS_PP
echo "Validation FlowSeek_chairs (S)"
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_C_1.pth_20251101124756 --dataset sintel --fsPP $FS_PP
python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_C_1.pth_20251101124756 --dataset kitti  --fsPP $FS_PP
echo "Validation FlowSeek_chairs (M)"
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_C_"$FS_PP".pth --dataset sintel   --fsPP $FS_PP
python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_C_"$FS_PP".pth --dataset kitti    --fsPP $FS_PP
echo "Validation FlowSeek_chairs (L)"
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_C_"$FS_PP".pth --dataset sintel   --fsPP $FS_PP
python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_C_"$FS_PP".pth --dataset kitti    --fsPP $FS_PP


# echo "TABLE 3 (CT)"
# echo "Validation FlowSeek (T)"
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_CT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_CT.pth --dataset kitti

# echo "Validation FlowSeek (S)"
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_CT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_CT.pth --dataset kitti

# echo "Validation FlowSeek (M)"
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_CT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_CT.pth --dataset kitti

# echo "Validation FlowSeek (L)"
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_CT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_CT.pth --dataset kitti

# echo "TABLE 3 (Tartan+CT)"
# echo "Validation FlowSeek (T)"
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT.pth --dataset kitti

# echo "Validation FlowSeek (S)"
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT.pth --dataset kitti

# echo "Validation FlowSeek (M)"
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT.pth --dataset kitti

# echo "Validation FlowSeek (L)"
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset sintel
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT.pth --dataset kitti

# echo "TABLE 4"
# echo "Validation FlowSeek (T)"
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --scale -1
# echo "Validation FlowSeek (S)"
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset spring --scale -1

# echo "Validation FlowSeek (M)"
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset spring --scale -1
# echo "Validation FlowSeek (L)"
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset spring --scale -1

# echo "TABLE 5 + TABLE B (SUPPLEMENTARY)"
# echo "Validation FlowSeek (T)"
# python evaluate.py --cfg config/eval/flowseek-T.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset layeredflow
# echo "Validation FlowSeek (S)"
# python evaluate.py --cfg config/eval/flowseek-S.json --model weights/flowseek_T_TartanCT_TSKH.pth --dataset layeredflow

# echo "Validation FlowSeek (M)"
# python evaluate.py --cfg config/eval/flowseek-M.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset layeredflow
# echo "Validation FlowSeek (L)"
# python evaluate.py --cfg config/eval/flowseek-L.json --model weights/flowseek_M_TartanCT_TSKH.pth --dataset layeredflow