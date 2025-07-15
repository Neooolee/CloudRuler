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
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return torch.mean((src - dst)**2)
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
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
def get_write_label(row_list): #get_write_picture函数得到训练过程中的可视化结果
    row_=[]    
    for i in range(len(row_list)):
        row=row_list[i] 
        col_=[]
        for j in range(len(row)):
            x_image=row[j]
            if j<1:
                x_image=x_image[:,:,[2,1,0]]
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

def read_labels(datapath,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=imgread(datapath[i])
        img=randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    imgs = rgb_to_gray(imgs) 
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
        # img = np.concatenate([img[:,:,:-2],img[:,:,-2:]],axis=-1)
        img = np.ascontiguousarray(randomflip(img,self.rn_list[index]))
        data["img"]=img.astype(np.float32)

        return data

    def __len__(self):
        return len(self.file_list)
class iterate_label(Dataset):
    def __init__(self,file_list, rn_list=None):
        self.file_list = file_list
        self.rn_list=rn_list
        # if rn_list==None:
        #     self.rn_list= [2 for _ in range(len(file_list))]#如果不指定翻转参数，就不翻转
        # else:
        #     self.rn_list=rn_list

    def __getitem__(self, index):
        path = self.file_list[index]
        img = read_img(path)
        img = randomflip(img,self.rn_list[index])
        img = np.uint8(img>0.8)

        return img

    def __len__(self):
        return len(self.file_list)

rgb_colors=OrderedDict([
    ("cloud-free",np.array([0],dtype=np.uint8)),
    ("cloud",np.array([255],dtype=np.uint8))])
gray_colors=OrderedDict([
    ("cloud-free",np.array([0,0,0],dtype=np.uint8)),
    ("cloud",np.array([255,255,255],dtype=np.uint8))])
#输入shape=(w,h)/(batch_size,w,h)/(batch_size,w,h,c)
def rgb_to_gray(rgb_mask):
    label = (np.zeros(rgb_mask.shape[:3]+tuple([1]))).astype(np.uint8)
    if len(rgb_mask.shape)==4:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((rgb_mask == np.asarray(rgb_values)).astype(np.uint8).sum(-1) == 3)
            label[match_pixs] = gray        
    else:
        for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
            match_pixs = np.where((rgb_mask == np.asarray(rgb_values)).astype(np.uint8) == 1)
            label[match_pixs] = gray
    return label.astype(np.uint8)

#输入shape=(w,h,c)/(batch_size,w,h,c)
def label_to_rgb(labels):
    max_index=np.argmax(labels,axis=-1)#第三维上最大值的索引，返回其他维度，并在并对位置填上最大值之索引
    n=len(labels.shape)-1
    if labels.shape[-1]<3:
        rgb = (np.zeros(labels.shape[:n])).astype(np.uint8)
    else:
        rgb = (np.zeros(labels.shape[:n]+tuple([3]))).astype(np.uint8)
    for gray, (class_name,rgb_values) in enumerate(rgb_colors.items()):
        match_pixs = np.where(max_index == gray)    
        rgb[match_pixs] = rgb_values
    return rgb.astype(np.uint8)

# def down_sample(input_,kernel_size,classes):
#     onehot=torch.one_hot(input_,classes)
#     onehot=nn.AveragePooling2D((kernel_size,kernel_size),kernel_size,'same')(onehot)
#     onehot=torch.argmax(onehot,axis=-1)
#     return onehot
# l=down_sample(np.ones((1,4,4,3)),2,2)
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
    
def diydecay(steps,baselr,cycle_step=100000,decay_steps=100,decay_rate=0.98,cirle_rate=0.98):
    n=steps//cycle_step
    clr=baselr*(cirle_rate**n)  
    steps=steps-n*cycle_step
    k=steps//decay_steps
    dlr = clr*(decay_rate**k) 
    return dlr
def cos_decay(steps,baselr,cycle_step=100000,decay_steps=100,cirle_rate=0.5):
    n=steps//cycle_step
    clr=baselr*(cirle_rate**n)  
    steps=steps-n*cycle_step
    k=steps//decay_steps
    t=cycle_step//decay_steps
    cos_lr = clr*((1+np.cos(k/t*np.pi))*0.5)  
    return cos_lr
def sin_decay(steps,baselr,minlr=1e-2,total_step=1e10,decay_steps=100):
    gama=(steps//decay_steps)/(total_step//decay_steps)*np.pi
    sin_lr = baselr*((1+np.cos(gama))*0.5)
    return np.where(sin_lr>minlr,sin_lr,minlr)
def warm_up(steps,baselr,minlr=1e-2,warm_end_step=10000,decay_steps=100):
    gama=((steps//decay_steps)/(warm_end_step//decay_steps)-1)*0.45*np.pi
    sin_lr = baselr*np.cos(gama)
    return np.where(sin_lr>minlr,sin_lr,minlr)
def lr_decay(global_steps,baselr,start_decay_step=10000,minlr=1e-2,total_step=1e10,decay_steps=100):
    lr=np.where(np.greater_equal(global_steps,start_decay_step),
                sin_decay(global_steps-start_decay_step,baselr,minlr,total_step-start_decay_step,decay_steps),baselr)
                # warm_up(global_steps,baselr,minlr,start_decay_step,decay_steps))
    return lr

def grad(src):
    g_src_x = src[:, :, 1:, :] - src[:, :, :-1, :]
    g_src_y = src[:, :, :, 1:] - src[:, :, :, :-1]
    return g_src_x,g_src_y
def all_comp(grad1,grad2):
    v=[]
    dim1=grad1.shape[-1]
    dim2=grad2.shape[-1]
    for i in range(dim1):
        for j in range(dim2):
            v.append(torch.mean(((grad1[:,i,:,:]**2)*(grad2[:,j,:,:]**2)))**0.25)
    return v

def get_grad(src,dst,level):
    gradx_loss=[]
    grady_loss=[]
    for i in range(level):
        gradx1,grady1=grad(src)
        gradx2,grady2=grad(dst)
        # lambdax2=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))
        # lambday2=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))
        lambdax2=1
        lambday2=1
        kapa=2
        gradx1_s=kapa*gradx1
        grady1_s=kapa*grady1
        gradx2_s=kapa*gradx2
        grady2_s=kapa*grady2
        gradx_loss+=all_comp(gradx1_s,gradx2_s)
        grady_loss+=all_comp(grady1_s,grady2_s)
        src=nn.functional.AvgPool2d(src,(2,2))
        dst=nn.functional.AvgPool2d(dst,(2,2))
    return gradx_loss,grady_loss

def exlusion_loss(src,dst,level=3):
    dim1=src.shape[-1]
    dim2=dst.shape[-1]
    gradx_loss,grady_loss=get_grad(src,dst,level)
    loss_gradxy=sum(gradx_loss)/(level*dim1*dim2)+sum(grady_loss)/(level*dim1*dim2)
    return loss_gradxy/2.0

# def gradxy(src):
#     src = torch.pad(src,[[0,0],[1,0],[1,0],[0,0]],mode="SYMMETRIC")#在行和列前各填充一行一列0
#     I_x = src[:,1:,1:,:]-src[:,1:,:-1,:]
#     I_y = src[:,1:,1:,:]-src[:,:-1,1:,:]
#     return I_x,I_y
# def grad_map(src):
#     I_x,I_y = gradxy(src)
#     return tf.math.sqrt(tf.math.square(I_x)+tf.math.square(I_y)+1e-20)       

def smooth_loss(src):
    x,y=grad(src)    
    g_loss=np.mean(np.abs(x))+np.mean(np.abs(y))
    return g_loss
  