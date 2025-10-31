SIZE="T"
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

# 开始训练流程
python train.py --cfg config/train/C368x496-"$SIZE".json --savedir checkpoints/ --fsPP $FS_PP
# python train.py --cfg config/train/C-T432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_C.pth

# python train.py --cfg config/train/C-T-TSKH432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_CT.pth
# python train.py --cfg config/train/Tartan480x640-"$SIZE".json --savedir checkpoints/
# python train.py --cfg config/train/Tartan-C368x496-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_Tartan.pth
# python train.py --cfg config/train/Tartan-C-T432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_TartanC.pth
# python train.py --cfg config/train/Tartan-C-T-TSKH432x960-"$SIZE".json --savedir checkpoints/ --restore_ckpt checkpoints/flowseek_"$SIZE"_TartanCT.pth