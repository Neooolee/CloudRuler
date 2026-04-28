# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:08:04 2018

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
import numpy as np
import glob
from PIL import Image
from gdaldiy import *
from CloudRuler import *
from evaluate import *
from pathlib import Path
from thop import profile
parser = argparse.ArgumentParser(description='')
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
parser.add_argument("--x_test_data_path", default=r'H:\data\NUAAL8-CR\test\CLOUD', help="path of x test datas.") #x域的测试图片路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--bands", type=int, default=[4,6,3], help="load batch size") #batch_size
parser.add_argument("--batch_size", type=int, default=1, help="load batch size")
parser.add_argument("--snapshot_dir", default='./snapshots/',help="Path of Snapshots") #读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_out/',help="Output Folder") #保存x域的输入图片与生成的y域图片的路径
args = parser.parse_args()

def make_data_list(data_path,filetype="*.tif"): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filedirs=Path(data_path)
    return [str(filepath) for filepath in filedirs.rglob(filetype)]

def main(num):
    if not os.path.exists(args.out_dir): #如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir)      
    x_datalists= make_data_list(args.x_test_data_path) #得到待测试的x域和y域图像路径名称列表
    model = CloudRuler(indim=3,outdim=3,resolution=args.image_size,gfdim=48,window_size=8,dilate_rate=4)    

    model.to('cuda')
    model.load_state_dict(torch.load(args.snapshot_dir+"ckpt-"+str(num)+".pth",map_location=torch.device('cuda'))['state_dict'])
    model.eval()
    flops, params = profile(model, inputs=(torch.randn(1,3,256,256).to("cuda"),))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Fparams = ' + str(params/1000**2) + 'Million')
    print('开始处理',datetime.now())
    starttime=datetime.now()  
    for i in range(len(x_datalists)):
        out_path=x_datalists[i].split('\\')
        testx = imgread(x_datalists[i])[:,:,1:4].astype(np.float32)[np.newaxis,:]
        out_list=model(torch.from_numpy(testx).to('cuda').permute(0,3,1,2).clip(0,1)).permute(0,2,3,1).clip(0,1)   
        write_image=out_list.data.cpu().numpy()[0]
        savepath=args.out_dir+out_path[-2]
        if not os.path.exists(savepath): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(savepath)
        savepath=savepath+'/'+out_path[-1].split('.')[-2]+'.tif'
        imgwrite(savepath,np.float32(write_image))

    endtime=datetime.now()     
    print('结束时间：',endtime)
    deltatime=endtime-starttime
    print('检测用时:',deltatime) 
    print('平均用时',deltatime)
        
if __name__ == '__main__':
    ckpt="rgb"
    main(str(ckpt))
    compute_ps(ckpt)

