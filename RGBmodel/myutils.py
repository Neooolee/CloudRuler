# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""
import numpy as np
from torch.utils.data import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from gdaldiy import *
import os
from pathlib import Path
def make_data_list(data_path,filetype="*.tif"): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filedirs=Path(data_path)
    return [str(filepath) for filepath in filedirs.rglob(filetype)]
    
def l1_loss(src, dst,weight=None): #定义l1_loss
    if weight is None:
        return torch.mean(torch.abs(src-dst))
    else:
        return torch.mean(weight*torch.abs(src-dst))
def l2_loss(x):
    return torch.sqrt(torch.sum(x**2))

def linear_2(img,x=0,y=1):#2%线性拉伸,返回0~1之间的值
    low,high=np.nanpercentile(img,(2,98),axis=[x,y],keepdims=True)
    img=np.clip(img,low,high)
    return (img-low)/(high-low)

def get_write_picture(row_list): #get_write_picture函数得到训练过程中的可视化结果
    row_=[]    
    for i in range(len(row_list)):
        row=row_list[i] 
        col_=[]
        for image in row:
            x_image=image[:,:,[2,1,0]]
            x_image=linear_2(x_image)
            col_.append(x_image)
        row_.append(np.concatenate(col_,axis=1))
    if len(row_list)==1:
        output = np.concatenate(col_,axis=1)
    else:
        output = np.concatenate(row_, axis=0) #得到训练中可视化结果
    return output*255
  
def randomflip(input_,n):
    #生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n<0:
        return np.rot90(input_,n)
    elif -1<n<2:
        return np.flip(input_,n)
    else: 
        return input_
def read_img(datapath,scale=255):
    img=imgread(datapath)
    img[img>scale]=scale
    img=img/scale   
    return img

def read_imgs(datapath,scale=255,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=read_img(datapath[i],scale)
        img = randomflip(img,k)
        img = img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    return imgs

class iterate_img(Dataset):
    def __init__(self,file_list, rn_list=None,scale=1):
        self.file_list = file_list
        self.scale=scale
        self.rn_list=rn_list

        if self.rn_list is None:
            self.rn_list= [2 for _ in range(len(file_list))]#如果不指定翻转参数，就不翻转

    def __getitem__(self, index):
        data=dict()
        path=self.file_list[index]
        data["filename"] = path.replace("\\","/").split("/")[-2]+"-"+path.replace("\\","/").split("/")[-1][:-4]
        img = imgread(path)/self.scale
        img = np.ascontiguousarray(randomflip(img,self.rn_list[index]))
        data["img"]=img.astype(np.float32)

        return data

    def __len__(self):
        return len(self.file_list)

def network_parameters(model):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
def get_last_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir+'checkpoint.txt'):
            with open(ckpt_dir+"checkpoint.txt",'r') as f:
                lastckpt_path=f.readlines()[-1].strip("\n")
    else:
        print("no checkpointlog")
    return lastckpt_path

def load_ckpt(ckpt_dir,ckptname=None):
    if os.path.exists(ckpt_dir+'checkpoint.txt'):
        if ckptname==None:
            with open(ckpt_dir+"checkpoint.txt",'r') as f:
                lastckpt_path=f.readlines()[-1].strip("\n")
        else:
            lastckpt_path=ckpt_dir+"ckpt-"+str(ckptname)+".pkl"
        print("loading checkpoint '{}'".format(lastckpt_path))
        checkpoint = torch.load(lastckpt_path)
        print("sucessfully loading checkpoint")
        return checkpoint
    else:
        print("no checkpoint found at '{}'".format(ckpt_dir))
        return None

def save_ckpt(mode_dict,ckpt_dir):
    ckpt_path=ckpt_dir+"ckpt-{}-{}.pth".format(mode_dict['epoch'],mode_dict['step'])
    with open(ckpt_dir+"checkpoint.txt",'a+') as f:
        lastckpt_paths=f.readlines()
        l=''
        for ckpt in lastckpt_paths:
            l=l+ckpt
        if ckpt_path not in l:
            f.write(ckpt_path+"\n")
    torch.save(mode_dict,ckpt_path)
    
def sin_decay(steps,baselr,minlr=1e-2,total_step=1e10,decay_steps=100):
    gama=(steps//decay_steps)/(total_step//decay_steps)*np.pi
    sin_lr = baselr*((1+np.cos(gama))*0.5)
    return np.where(sin_lr>minlr,sin_lr,minlr)

def lr_decay(global_steps,baselr,start_decay_step=10000,minlr=1e-2,total_step=1e10,decay_steps=100):
    lr=np.where(np.greater_equal(global_steps,start_decay_step),
                sin_decay(global_steps-start_decay_step,baselr,minlr,total_step-start_decay_step,decay_steps),baselr)
    return lr
