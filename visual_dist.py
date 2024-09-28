#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: train.py 
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""
import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime
import numpy as np

from Generation.model_test import Model
from Generation.config import opts
from train_dist import SPGAN
import random
from torch.autograd import Variable
import torch
import useful_func as uf
from Generation.point_operation import plot_pcd_multi_rows,plot_pcd_multi_rows_single_color,plot_pcd_three_views_color,plot_pcd_multi_rows_color
import time
from Generation.point_operation import normalize_point_cloud

plot_folder = "models/plots"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

def pc_normalize(pc,return_len=False):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    return pc

class visualization(object):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.model = SPGAN(opts)
        checkpoint = torch.load('./pretrain_checkpoints/' +opts.choice+'.ckpt', map_location='cuda:0') 
        self.model.load_state_dict(checkpoint['state_dict'])
        self.G = self.model.G.to('cuda:0')

        
    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))
                # scale = self.opts.nv
                # w = np.random.uniform(low=-scale, high=scale, size=(bs, 1, self.opts.nz))
                noise = np.tile(noise, (1, self.opts.np, 1))

            if self.opts.n_mix and random.random() < 0.8:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    idx = np.arange(self.opts.np)
                    np.random.shuffle(idx)
                    num = int(random.random() * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise
    
    def sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt('template/4096.xyz')[:, :3]
            else:
                self.ball = np.loadtxt('template/ball2.xyz')[:, :3]
            self.ball = pc_normalize(self.ball)

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball
    
    def half_sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt('template/4096.xyz')[:, :3]
            else:
                self.ball = np.loadtxt('template/ball2.xyz')[:, :3]
            self.ball = pc_normalize(self.ball)

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball
    
    def read_ball(self,sort=False):
        x = np.loadtxt("template/balls/4096.xyz")
        # x = np.loadtxt("template/sphere/4096.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]

        # order = np.argsort(dist[1000])[::1]
        # ball = ball[order]

        return ball

    def draw_correspondense(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)
        ball = np.expand_dims(ball,axis=0)

        # self.build_model_eval()

        cat = str(self.opts.choice).lower()
        # could_load, save_epoch = self.load(self.opts.log_dir)
        # if could_load:
        #     start_epoch = save_epoch
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")
        #     exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")


        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 6


        number = 6 #z.shape
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        for i in range(sample_num):

            noise = np.random.normal(0, 0.2, (number, self.opts.nz)) #nz 128

            color = np.zeros((number+1, self.opts.np, 3))
            color_c = np.squeeze(ball,axis=0)
            color_c = np.minimum(color_c,1.0)
            color_c = np.maximum(color_c,-1.0)

            for j in range(color.shape[0]):
                color[j] = color_c
            title = ["Raw"]
            title.extend(["Sample_%d"% num for num in range(number)])

            noise = np.expand_dims(noise,axis=1)
            noise = np.tile(noise, (1, self.opts.np, 1))

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()

                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2,1)

            sample_pcs = out_pc.cpu().detach().numpy()

            sample_pcs = normalize_point_cloud(sample_pcs)
            pcds = np.concatenate([0.5*ball,0.75*sample_pcs],axis=0)

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_correspondense_%s_%d.png" % (current_time, i))
            uf.plot_point_cloud(sample_pcs[0], save_fig=True, plot_name=plot_name)

            plot_pcd_three_views_color(plot_name,pcds,title,colors=color)

        del self.G


if __name__ == '__main__':


    # opts.pretrain_model_G = "300_cookie_G.pth"
    # opts.log_dir = "models"

    model = visualization(opts)

    model.draw_correspondense() # draw the correspondense between sphere and shape
    #model.draw_shape_intepolate() # shape inteporlate
    #model.draw_part_shape_inte()  # shape inteporlate vs part-wise shape inteporlate
    #model.draw_part_shape_inte_detail() # shape inteporlate vs multi-path part-wise shape inteporlate

    #model.draw_part_edit() # random change the noise on selected region
    #model.draw_part_flip() # negative the noise vector along x,y,z zxis
    #model.draw_edit_inte() # combine for part edit & part/shape interpolate

    #model.draw_part_exchange() # exchange the noise vector of two regions of two shape