import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import logging
import random
from Generation.Generator import Generator
from Generation.Discriminator import Discriminator
from Common.loss_utils import get_local_pair,compute_all_metrics2,AverageValueMeter,dist_simple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
from Generation.config import opts
from Common.network_utils import *
from Common import loss_utils
from Generation.H5DataLoader import H5DataLoader
import os
from datetime import datetime
from data.train_mvtec_dataset import get_data_loader
import useful_func as uf

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class SPGAN(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.automatic_optimization = False
        self.G = Generator(self.opts)
        self.D = Discriminator(self.opts)
        self.ball = None
        self.knn_loss = uf.kNNLoss(k=self.opts.knn_k,n_seeds=self.opts.knn_n_seeds)
    
    def configure_optimizers(self):
        beta1 = 0.5
        beta2 = 0.99
        optimizerG = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.opts.lr_g, betas=(beta1, beta2))
        optimizerD = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.opts.lr_d, betas=(beta1, beta2))
        if self.opts.lr_decay:
            if self.opts.use_sgd:
                self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, self.opts.max_epoch, eta_min=self.opts.lr_g)
            else:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=self.opts.lr_decay_feq, gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_G = None

        if self.opts.lr_decay:
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=self.opts.lr_decay_feq, gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_D = None
            
        if not self.opts.lr_decay:
             return [optimizerG, optimizerD], []
        else:  
            return [optimizerG, optimizerD], [self.scheduler_G, self.scheduler_D]

    def forward(self, x, z):

        return self.G(x, z)
    
    def on_train_epoch_start(self):
        self.epoch_g_loss = []
        self.epoch_d_loss = []

    def on_train_epoch_end(self):
        # Get the current epoch
        epoch = self.current_epoch    
        if self.scheduler_G is not None:
            self.scheduler_G.step(epoch)
        if self.scheduler_D is not None:
            self.scheduler_D.step(epoch)    
        d_loss_mean = np.array(self.epoch_d_loss).mean()
        g_loss_mean = np.array(self.epoch_g_loss).mean()
        self.log('G_loss', g_loss_mean, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('D_loss', d_loss_mean, on_step=False, on_epoch=True,sync_dist=True, prog_bar=True, logger=True) 


    def training_step(self, batch):
        sample, _ = batch
        data = sample[1]
        #-------------------
        # data = batch
        optimizerG, optimizerD  = self.optimizers()
        if opts.lr_decay:
            schedulerG, schedulerD = self.lr_schedulers()
        else:
            schedulerG, schedulerD = None, None

        x = self.sphere_generator(bs=self.opts.bs)

        for d_itera in range(self.opts.D_itera):
            self.D.zero_grad()
            real_points = Variable(data, requires_grad=False)
            # requires_grad(self.D, True)
            # requires_grad(self, False)
            z = self.noise_generator(bs=self.opts.bs)
            d_fake_preds =self.G(x, z)
            real_points = real_points.transpose(2, 1)
            d_fake_preds = d_fake_preds.detach()
            d_real_logit = self.D(real_points)
            d_fake_logit = self.D(d_fake_preds)
            lossD, info = loss_utils.dis_loss(d_real_logit,d_fake_logit,gan=self.opts.gan,noise_label=self.opts.flip_d)
            self.manual_backward(lossD)
            optimizerD.step()
            optimizerD.zero_grad()
        
        self.epoch_d_loss.append(lossD.item())
        # requires_grad(self.D, False)
        # requires_grad(self, True)
        self.zero_grad()
        z = self.noise_generator(bs=self.opts.bs)
        g_fake_preds =self.G(x, z)
        g_real_logit = self.D(real_points)
        g_fake_logit = self.D(g_fake_preds)
        
        # knn_loss = self.knn_loss(g_fake_preds.transpose(2,1))
        lossG,_ = loss_utils.gen_loss(g_real_logit,g_fake_logit,gan=self.opts.gan,noise_label=self.opts.flip_g)
        # lossG += knn_loss

        self.manual_backward(lossG)
        optimizerG.step()
        self.zero_grad()
        self.epoch_g_loss.append(lossG.item())

    def noise_generator(self, bs=1,masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))
                #scale = self.opts.nv
                #w = np.random.uniform(low=-scale, high=scale, size=(bs, 1, self.opts.nz))
                noise = np.tile(noise,(1,self.opts.np,1))

            if self.opts.n_mix and random.random() < 0.5:
               noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
               for i in range(bs):
                   id = np.random.randint(0,self.opts.np)
                   idx = np.argsort(self.ball_dist[id])[::1]
                   # idx = np.arange(self.opts.np)
                   # np.random.shuffle(idx)
                   num = int(max(random.random(),0.1)*self.opts.np)
                   noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i,idx] = idx

        sim_noise = Variable(torch.Tensor(noise), requires_grad=False).cuda()

        return sim_noise

    def half_sphere_generator(self,bs=2,static=True):

        if self.ball is None:
            self.ball = np.loadtxt('template/sphere/%d.xyz'%self.opts.np)[:,:3]
            self.ball = pc_normalize(self.ball)

            N = self.ball.shape[0]
            # xx = torch.bmm(x, x.transpose(2,1))
            xx = np.sum(self.ball ** 2, axis=(1)).reshape(N, 1)
            yy = xx.T
            xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
            self.ball_dist = xy + xx + yy  # [B, N, N]

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball), requires_grad=False).cuda()

        return ball

    def sphere_generator(self,bs=2,static=True):

        if self.ball is None:
            self.ball = np.loadtxt('template/balls/%d.xyz'%self.opts.np)[:,:3]
            self.ball = pc_normalize(self.ball)

            N = self.ball.shape[0]
            # xx = torch.bmm(x, x.transpose(2,1))
            xx = np.sum(self.ball ** 2, axis=(1)).reshape(N, 1)
            yy = xx.T
            xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
            self.ball_dist = xy + xx + yy  # [B, N, N]

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball), requires_grad=False).cuda()

        return ball
    
    
        
if __name__ == '__main__':
    _, train_loader = get_data_loader("train", class_name=opts.choice, img_size=224, batch_size=opts.bs)
    torch.set_float32_matmul_precision('high')
    checkpoint_callback = ModelCheckpoint(
    dirpath='./pretrain_checkpoints/',
    filename=opts.choice,
    save_weights_only=True
    )
    model = SPGAN(opts)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=[1,2,3],
                         strategy=DDPStrategy(find_unused_parameters=True),
                         log_every_n_steps=1,
                         min_epochs=1,
                         max_epochs=700,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader)
    