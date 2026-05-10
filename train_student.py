import collections
import collections.abc

# 这里的“黑魔法”是为了让 Python 3.10+ 兼容老代码的写法
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from config.nyu_b4 import config
from dataloader.dataloader_nyu import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from models.builder import EncoderDecoder3 as segmodel3
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, parse_devices
from utils.metric import hist_info, compute_score
from engine.evaluator import Evaluator
from utils.visualize import print_iou

# 命令行参数 (保留原作者的 parser 防报错，但实际训练不再需要 KD 参数)
parser = argparse.ArgumentParser()
parser.add_argument('--distillation_alpha', type=float, default=1)
parser.add_argument('--distillation_beta', type=float, default=0.1)
parser.add_argument('--distillation_single', type=int, default=1)
parser.add_argument('--mask_single', type=str, default='hint')
parser.add_argument('--distillation_flag', type=int, default=0)
parser.add_argument('--lambda_mask', type=float, default=0.75)
parser.add_argument('--select', type=str, default='max')
parser.add_argument('--decode_init', type=int, default=0)
parser.add_argument('--losses', nargs='+', default=['loss1','loss2','loss3','loss4'])
logger = get_logger()
os.environ['MASTER_PORT'] = '169710'

# ================= 原作者的评估器 =================
class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, flag):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        if flag == "rgb":
            pred = self.sliding_eval_rgbX(img, None, config.eval_crop_size, config.eval_stride_rate, device)
        elif flag == 'depth':
            pred = self.sliding_eval_rgbX(modal_x, None, config.eval_crop_size, config.eval_stride_rate, device)
        else:
            pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line, mIoU = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                      config.class_names, show_no_back=False)
        return result_line, mIoU

class Record(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# ================= 核心主程序 =================
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    val_pre = ValPre()
    val_dataset = RGBXDataset(data_setting, 'val', val_pre)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M-%S", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
        path3 = tb_dir + '/exp.log'
        sys.stdout = Record(path3, sys.stdout)
    
    print("\n========== [纯净版] 仅训练学生模型 (RGB+HHA) ==========\n")

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # 初始化学生模型 (去除了单通道骨干的覆盖逻辑)
    if args.mask_single == "mask_hint":
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, load=True, decode_init=0, losses=args.losses, lambda_mask=args.lambda_mask)
    else:
        model = segmodel3(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, load=True, decode_init=0, losses=args.losses, lambda_mask=args.lambda_mask)

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # 极速优化：Channels Last (完美契合 V100 AMP)
        #model = model.to(memory_format=torch.channels_last)

    # 注册引擎状态 (删除了 model2 和 optimizer2)
    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    logger.info('begin training:')
    
    # 极速优化：AMP 缩放器
    scaler = torch.cuda.amp.GradScaler()

    Best_IoU = 0.0

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        
        sum_loss = 0
        
        for idx in pbar:
            engine.update_iteration(epoch, idx)
            # 兼容高版本 Python 的迭代写法
            minibatch = next(dataloader) 
            imgs = minibatch['data'].cuda(non_blocking=True)
            gts = minibatch['label'].cuda(non_blocking=True)
            modal_xs = minibatch['modal_x'].cuda(non_blocking=True)
            # 🔍 强制安检：检查标签是否越界
            unique_labels = torch.unique(gts)
            max_label = unique_labels.max().item()
            # 假设 config.num_classes 是 40，那么合法的索引是 0~39，或者是 255 (ignore)
            if max_label >= config.num_classes and max_label != 255:
                print(f"\n❌ [致命错误] 发现非法标签值: {max_label}!")
                print(f"当前合法范围是 0 到 {config.num_classes - 1}，或者 255。")
                print(f"当前批次中包含的所有类别索引为: {unique_labels.tolist()}")
                # 强制修正：把越界的值暂时设为 255，防止崩溃（但这只是临时方案，你需要检查数据预处理）
                gts[gts >= config.num_classes] = 255
            # 开启自动混合精度结界
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 仅学生前向传播
                logits, rgbd_x, loss = model(imgs, modal_xs, gts)

            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            else:
                reduce_loss = loss

            # 混合精度反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += reduce_loss.item()
            print_str = f"Epoch {epoch}/{config.nepochs} Iter {idx + 1}/{config.niters_per_epoch}: lr={lr:.4e} loss={reduce_loss.item():.4f} avg_loss={(sum_loss / (idx + 1)):.4f}"

            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            
        # ================= 验证环节 =================
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            # 清理训练残留显存
            torch.cuda.empty_cache()
            model.eval()
            
            device_str = '0'
            all_dev = parse_devices(device_str)
            with torch.no_grad():
                segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
                                         config.norm_std, None,
                                         config.eval_scale_array, config.eval_flip,
                                         all_dev, verbose=False, save_path=None, show_image=False)
                
                config.val_log_file = tb_dir + '/val_' + '.log'
                config.link_val_log_file = tb_dir + '/val_last.log'
                config.checkpoint_dir = tb_dir + '/checkpoint'
                
                # 修复原版单卡丢失 "rgbd" 参数的 Bug
                mIoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                     config.link_val_log_file, model, "rgbd")
                
                print('\n>>> epoch: %d, mIoU: %.3f%%, Best_IoU: %.3f%%\n' % (epoch, mIoU, Best_IoU))
                
                if (Best_IoU < mIoU):
                    Best_IoU = mIoU
                    engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link, Best_IoU)
                    print("save successful!")