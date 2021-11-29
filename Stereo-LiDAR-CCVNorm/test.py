"""
Testing process.

Usage:
# For KITTI Depth Completion
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2017 --rgb_dir ./data/kitti2017/rgb --depth_dir ./data/kitti2015/depth
# For KITTI Stereo
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2015 --root_dir ./data/kitti_stereo/data_scene_flow
"""

import os
import sys
import time
import argparse
import importlib
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from misc import utils
from misc import metric
from dataset.dataset_kitti2017 import DatasetKITTI2017
from dataset.dataset_kitti2015 import DatasetKITTI2015


DISP_METRIC_FIELD = ['err_3px', 'err_2px', 'err_1px', 'rmse', 'mae']
DEPTH_METRIC_FIELD = ['rmse', 'mae', 'mre', 'irmse', 'imae']

SEED = 100
random.seed(SEED)
np.random.seed(seed=SEED)
cudnn.benchmark = False #寻找最适合当前配置的算法，但每次前馈会有差别，后续设置deterministic
cudnn.deterministic = True #每次返回的卷积算法将是确定的，保证每次运行网络相同输入的输出是相同的
torch.manual_seed(SEED)#torch.manual_seed(seed) 设定生成随机数的种子，并返回一个 torch._C.Generator 对象. 参数: seed (int or long) – 种子,为当前GPU设置种子
torch.cuda.manual_seed_all(SEED)#为所有GPU设置种子


def parse_arg():#参数读取模块
    parser = argparse.ArgumentParser(description='Sparse-Depth-Stereo testing')#创建对象，开头描述为‘’的参数文件
    parser.add_argument('--model_cfg', dest='model_cfg', type=str, default=None,
                        help='Configuration file (options.py) of the trained model.')#添加程序参数信息，模型结构
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Path to weight of the trained model.')#模型结构存储路径
    parser.add_argument('--dataset', dest='dataset', type=str, default='kitti2017',
                        help='Dataset used: kitti2015 / kitti2017')#数据集存储路径
    parser.add_argument('--rgb_dir', dest='rgb_dir', type=str, default='./data/kitti2017/rgb',
                        help='Directory of RGB data for kitti2017.')#rgb文件夹
    parser.add_argument('--depth_dir', dest='depth_dir', type=str, default='./data/kitti2017/depth',
                        help='Directory of depth data for kitti2015.')#depth文件夹
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='./data/kitti_stereo/data_scene_flow',
                        help='Root directory for kitti2015')#root文件夹
    parser.add_argument('--random_sampling', dest='random_sampling', type=float, default=None,
                        help='Perform random sampling on ground truth to obtain sparse disparity map; Only used in kitti2015')#数据集中随机采样
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true',
                        help='Don\'t use gpu')#检测cuda加速是否可用
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()#解析参数，检查命令行，转换参数为适当类型而后调用相关操作
    return args


def main():
    # Parse arguments
    args = parse_arg()#读取参数

    # Import configuration file
    sys.path.append('/'.join((args.model_cfg).split('/')[:-1]))#当前目录+模型参数文件存放目录，以/分隔
    options = importlib.import_module(((args.model_cfg).split('/')[-1]).split('.')[0])#从路径中导入对象
    cfg = options.get_config()#获得参数

    # Define model and load
    model = options.get_model(cfg.model_name)#获得模型
    if not args.no_cuda:#如果cuda可用
        model = model.cuda()
    train_ep, train_step = utils.load_checkpoint(model, None, None, args.model_path, True)

    # Define testing dataset (NOTE: currently using validation set)
    if args.dataset == 'kitti2017':
        dataset = DatasetKITTI2017(args.rgb_dir, args.depth_dir, 'my_test',
                                   (256, 1216), to_disparity=cfg.to_disparity, # NOTE: set image size to 256x1216
                                   fix_random_seed=True)
    elif args.dataset == 'kitti2015':
        dataset = DatasetKITTI2015(args.root_dir, 'training', (352, 1216), # NOTE: set image size to 352x1216
                                   args.random_sampling, fix_random_seed=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                        num_workers=4)

    # Perform testing
    model.eval()#测试模式，用于通知dropout层和batchnorm层在train和val模式切换
    #在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); 
    # batchnorm层会继续计算数据的mean和var等参数并更新。
    #在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean
    # 和var，直接使用在训练阶段已经学出的mean和var值
    #对比with torch.no_grad()主要停止autograd模块工作，以起到加速、节省显存的作用，停止gradient
    # 计算，不影响dropout和batchnorm运算
    pbar = tqdm(loader)
    pbar.set_description('Testing')
    disp_meters = metric.Metrics(DISP_METRIC_FIELD)#metric.py定义
    disp_avg_meters = metric.MovingAverageEstimator(DISP_METRIC_FIELD)
    depth_meters = metric.Metrics(DEPTH_METRIC_FIELD)
    depth_avg_meters = metric.MovingAverageEstimator(DEPTH_METRIC_FIELD)
    infer_time = 0
    with torch.no_grad():#在dropout和batchnorm失效的情况下，
        # 进一步将autograd默认的true置为false，保证反向过程不改变参数
        for it, data in enumerate(pbar):
            # Pack data
            if not args.no_cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()#调用cuda
            inputs = dict()#词典类型
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            if cfg.to_disparity:#若参数定义要求转换为视差图
                inputs['left_sd'] = data['left_sdisp']
                inputs['right_sd'] = data['right_sdisp']
            else:
                inputs['left_sd'] = data['left_sd']
                inputs['right_sd'] = data['right_sd']
            if args.dataset == 'kitti2017':
                target_d = data['left_d']
            target_disp = data['left_disp']
            img_w = data['width'].item()

            # Inference
            end = time.time()
            pred = model(inputs)
            if cfg.to_disparity:
                pred_d = utils.disp2depth(pred, img_w)
                pred_disp = pred
            else:
                raise NotImplementedError
            infer_time += (time.time() - end)

            # Measure performance
            if cfg.to_disparity:
                # disparity
                pred_disp_np = pred_disp.data.cpu().numpy()
                target_disp_np = target_disp.data.cpu().numpy()
                disp_results = disp_meters.compute(pred_disp_np, target_disp_np)
                disp_avg_meters.update(disp_results)
                if args.dataset == 'kitti2017':
                    # depth
                    pred_d_np = pred_d.data.cpu().numpy()
                    target_d_np = target_d.data.cpu().numpy()
                    depth_results = depth_meters.compute(pred_d_np, target_d_np)
                    depth_avg_meters.update(depth_results)
            else:
                raise NotImplementedError
    infer_time /= len(loader)

    if cfg.to_disparity:
        disp_avg_results = disp_avg_meters.compute()
        print('Disparity metric:')
        for key, val in disp_avg_results.items():
            print('- {}: {}'.format(key, val))
    if args.dataset == 'kitti2017':
        depth_avg_results = depth_avg_meters.compute()
        print('Depth metric:')
        for key, val in depth_avg_results.items():
             print('- {}: {}'.format(key, val))
    print('Average infer time: {}'.format(infer_time))


if __name__ == '__main__':
    main()
