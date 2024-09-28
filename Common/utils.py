#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: utils.py 
@time: 2020/10/27
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LRScheduler(object):

    def __init__(self, optimizer):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer

    def update(self, learning_rate, ratio=1):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = learning_rate * ratio**i

class DiscriminatorLoss(object):
    """
    feature distance from discriminator
    """
    def __init__(self, data_parallel=False):
        self.l2 = nn.MSELoss()
        self.data_parallel = data_parallel

    def __call__(self, D, fake_pcd, real_pcd):
        if self.data_parallel:
            with torch.no_grad():
                real_feature = nn.parallel.data_parallel(
                    D, real_pcd.detach())
            fake_feature = nn.parallel.data_parallel(D, fake_pcd)
        else:
            with torch.no_grad():
                real_feature = D(real_pcd.detach())
            fake_feature = D(fake_pcd)

        D_penalty = F.l1_loss(fake_feature, real_feature)
        return D_penalty
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
import math
import sys
from numbers import Number
from collections import Set, Mapping, deque

import torch
import numpy as np
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from ChamferDistancePytorch import fscore

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res

def calc_dcd_full(x, gt, T=1000, n_p=1, return_raw=False, separate=False, return_freq=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True, separate=separate)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * T), torch.exp(-dist2 * T)

    loss1 = []
    loss2 = []
    gt_counted = []
    x_counted = []

    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_p
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_p
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

        if return_freq:
            expand_count1 = torch.zeros_like(idx2[b])  # n_x
            expand_count1[:count1.shape[0]] = count1
            x_counted.append(expand_count1)
            expand_count2 = torch.zeros_like(idx1[b])  # n_gt
            expand_count2[:count2.shape[0]] = count2
            gt_counted.append(expand_count2)

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    if separate:
        res = [torch.cat([loss1.unsqueeze(0), loss2.unsqueeze(0)]), cd_p, cd_t]
    else:
        res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    if return_freq:
        x_counted = torch.stack(x_counted)
        gt_counted = torch.stack(gt_counted)
        res.extend([x_counted, gt_counted])
    return res