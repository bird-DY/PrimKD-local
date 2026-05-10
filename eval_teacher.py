import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# 导入配置和数据加载
from config.nyu_b4 import config
from dataloader.dataloader_nyu import ValPre
from dataloader.RGBXDataset import RGBXDataset

# 只导入教师模型 (EncoderDecoder2)
from models.builder import EncoderDecoder2 as segmodel2

# 导入评估相关组件
from utils.pyt_utils import parse_devices
from utils.metric import hist_info, compute_score
from utils.visualize import print_iou
from engine.evaluator import Evaluator

# ========================================================
# 1. 复制原有的评估器类 (保持不变)
# ========================================================
class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, flag):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        
        # 因为测试的是老师，只会走到 flag == "rgb" 这个分支
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
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1
        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        
        # 注意：这里需要传入 class_names
        result_line, mIoU = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                      config.class_names, show_no_back=False)
        return result_line, mIoU

# ========================================================
# 2. 主测试流程
# ========================================================
if __name__ == '__main__':
    print("\n========== [开始体检] 正在初始化纯 RGB 教师模型 ==========")
    
    # 初始化验证数据集
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

    # 声明损失函数 (验证时其实用不到 backward，但构建模型需要)
    criterion2 = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    BatchNorm2d2 = nn.BatchNorm2d

    # 初始化教师模型 (自动加载 config 里的 pretrained_model1，即 nyu-rgb.pth)
    config.backbone = 'single_'+config.backbone
    model_teacher = segmodel2(cfg=config, criterion=criterion2, norm_layer=BatchNorm2d2, load=True, decode_init=1)
    
    # 转移到 GPU 并开启 eval 模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_teacher.to(device)
    model_teacher.eval()

    print("\n========== [加载完毕] 开始在测试集上验证 ==========")

    # 准备评估器
    all_dev = parse_devices('0')
    with torch.no_grad():
        segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network=None,
                                 scale_array=config.eval_scale_array, flip=config.eval_flip,
                                 devices=all_dev, verbose=False, save_path=None,
                                 show_image=False)
        
        # 定义临时日志路径
        config.val_log_file = './val_teacher_test.log'
        config.link_val_log_file = './val_teacher_last.log'
        config.checkpoint_dir = './'
        
        # 🚀 核心：执行测试！注意最后一个参数是 "rgb"，防止传入 HHA 数据
        mIoU = segmentor.run(config.checkpoint_dir, "Teacher", config.val_log_file,
                             config.link_val_log_file, model_teacher, "rgb")
        
        print('\n========== [体检结束] 教师模型最终 mIoU: %.3f%% ==========' % mIoU)