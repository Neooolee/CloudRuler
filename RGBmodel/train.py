# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:57:36 2018

@author: Neoooli
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import torch
from torch.autograd import Variable
from torch.utils.data import *
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import glob
from PIL import Image
from CloudRuler import *
from myutils import *
from gdaldiy import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='')
parser.add_argument("--snapshot_dir", default='./snapshots/', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=2e-4, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch') #训练的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=1000, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=1000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--y_train_data_path", default=r'H:\data\NUAAL8-CR\train\GT', help="path of y training datas.") #y域的训练图片路径
parser.add_argument("--batch_size", type=int, default=8,help="load batch size") #batch_size
args = parser.parse_args()

class maintrain(object):
    """docstring for maintrain"""
    def __init__(self,device):
        super(maintrain, self).__init__()
        self.device=device
        self.Net = CloudRuler(indim=3,outdim=3,resolution=args.image_size,gfdim=48)
        self.Net.to(device)
        self.Net.train()
        self.optimizer = torch.optim.Adam(self.Net.parameters(),lr=args.base_lr,betas=(args.beta1,args.beta2))

    def train_step(self,ximage_list,yimage_list,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
        
        ximages=ximage_list[0].to(self.device).permute(0,3,1,2).clip(0,1)[:,1:4]
        yimages=yimage_list[0].to(self.device).permute(0,3,1,2).clip(0,1)[:,1:4]
        fake_y_list=self.Net(ximages)

        fgx,fgy=grad(fake_y_list)
        gx,gy=grad(yimages)
        nll_loss=torch.sum(torch.mean(torch.abs(fake_y_list-yimages),dim=[0,2,3]))\
        +0.5*torch.sum(torch.mean(torch.abs(fgx-gx),dim=[0,2,3]))\
        +0.5*torch.sum(torch.mean(torch.abs(fgy-gy),dim=[0,2,3]))

        self.optimizer.zero_grad()
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss,nll_loss,fake_y_list.permute(0,2,3,1).clip(0,1) 
    def train(self,y_datalists,batch_size,lrbz):
        print ('Start Training')
        checkpoint=load_ckpt(args.snapshot_dir)
        start_epoch=1
        step=1
        if checkpoint is not None:
            self.Net.load_state_dict(checkpoint['state_dict'])
            start_epoch=checkpoint['epoch']
            step=checkpoint['step']
        writer=SummaryWriter(args.snapshot_dir)
        print(network_parameters(self.Net))
        leny=len(y_datalists)
        start_epoch=(step*batch_size)//leny+1
        scale=1
        total_step=args.epoch*leny//batch_size
        start_decay_step=leny//batch_size

        for epoch in range(start_epoch,args.epoch+1): #训练epoch数       
            #每训练一个epoch，就打乱一下x域图像顺序
            shuffle(y_datalists)

            x_datalists= [name.replace('GT','CLOUD') for name in y_datalists]

            k_list = np.random.randint(low=-3, high=3,size=leny)
            x_dataset=iterate_img(x_datalists,k_list,scale)
            x_dataset_loader=DataLoader(x_dataset,batch_size,num_workers=4,pin_memory=True,
                              drop_last=True)
            y_dataset=iterate_img(y_datalists,k_list,scale)
            y_dataset_loader=DataLoader(y_dataset,batch_size,num_workers=4,pin_memory=True,
                              drop_last=True)
            torch.cuda.empty_cache()
            for batch_inputx_img,batch_inputy_img in zip(x_dataset_loader,y_dataset_loader):
                lr=lr_decay(step,lrbz,start_decay_step=start_decay_step,minlr=lrbz*1e-2,total_step=total_step,decay_steps=10)
       
                gl,dl,fake_y= self.train_step([batch_inputx_img["img"]],[batch_inputy_img["img"]],lr) #得到每个step中的生成器和判别器loss
                step=step+1 
                if step% args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                    writer.add_scalar("loss",gl.data.cpu().numpy(),step)
                    writer.add_scalar("lr",self.optimizer.param_groups[0]['lr'],step)

                if step % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                    write_image = get_write_picture([[batch_inputx_img["img"].cpu().numpy()[0,:,:,1:4],
                            batch_inputy_img["img"].cpu().numpy()[0,:,:,1:4],
                            fake_y.data.cpu().numpy()[0]]]) #得到训练的可视化结果
                    write_image_name = args.out_dir + "/"+ str(epoch)+'_'+str(step)+"-"+batch_inputy_img["filename"][0]+ ".png" #待保存的训练可视化结果路径与名称
                    imgwrite(write_image_name,np.uint8(write_image)) #保存训练的可视化结果
                    print('epoch step     a_loss    d_loss    lr')
                    print('{:d}     {:d}    {:.3f}    {:.3f}    {:.8f} '.format(epoch,step,gl,dl,lr))
            if epoch%10 == 0:
                save_ckpt({'epoch':epoch,'step':step,'state_dict':self.Net.state_dict(),args.snapshot_dir)

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
        dtype=torch.FloatTensor
        torch.set_default_tensor_type(dtype)
    torch.autograd.set_detect_anomaly =True
    print('Using {} device'.format(device))
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')
    
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    y_datalists = make_data_list(args.y_train_data_path)
    batch_size=args.batch_size 
    lrbz=args.base_lr
    maintrain_object=maintrain(device)
    maintrain_object.train(y_datalists,batch_size,lrbz)
                
if __name__ == '__main__':
    main()

