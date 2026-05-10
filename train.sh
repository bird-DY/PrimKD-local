#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --port=29516 --distillation_alpha=1.0 --distillation_beta=0.1 --distillation_flag=1 --lambda_mask=0.75 --select="max" --mask_single="hint"
#A5000配置CUDA_VISIBLE_DEVICES=1 python train.py -d 0 --port=29516 --distillation_alpha=1.0 --distillation_beta=0.1 --distillation_flag=1 --lambda_mask=0.75 --select="max" --mask_single="hint"
#CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py -c /root/code/log_NYUDepthv2_mit_b4/tb/Apr25_25-22-54-25/checkpoint/epoch.pth -d 0 --port=29516 --distillation_alpha=1.0 --distillation_beta=0.1 --distillation_flag=1 --lambda_mask=0.75 --select="max" --mask_single="hint"

#学生单独训练配置
CUDA_VISIBLE_DEVICES=0 python train_student.py -d 0 --port=29516 --distillation_alpha=1.0 --distillation_beta=0.1 --distillation_flag=1 --lambda_mask=0.75 --select="max" --mask_single="hint"