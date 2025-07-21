# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:56:44 2018

@author: lijun
"""
import torch
import torch.nn as nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from modules import *

class patch_merge(nn.Module):
    def __init__(self,indim,outdim=None,ks=3,s=2):
        super().__init__()
        outdim= indim if outdim is None else outdim        
        self.down=nn.Conv2d(indim,outdim,ks,s,padding=1,padding_mode="reflect")
    def forward(self,x):
        x=self.down(x)
        return x
class patch_unmerge(nn.Module):
    def __init__(self,indim,outdim=None,scale_factor=2):
        super().__init__()
        outdim= indim if outdim is None else outdim
        self.scale_factor=scale_factor
        self.conv = torch.nn.Conv2d(indim,
                                    outdim,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,padding_mode="reflect")

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self,in_channels, out_channels=None,act=nn.ReLU()):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.act1=act
        self.act2=act
        self.norm1 = nn.GroupNorm(12,in_channels)
        self.norm2 = nn.GroupNorm(12,in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,padding_mode="reflect",groups=1)

        self.conv2 = nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,padding_mode="reflect",groups=1)

        if in_channels != out_channels:
            self.shortcutconv = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,padding_mode="reflect")

    def forward(self, x):
        h = x
        h= self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm1(h)
        h = self.act2(h)
        h = self.conv2(h)
        # h = self.norm2(h)
        
        if self.in_channels != self.out_channels:
            x=self.shortcutconv(x)
        return h+x

   
class CloudRuler(nn.Module):
    def __init__(self,indim=3,outdim=3,resolution=256,gfdim=24,ch_mult=[1,2,4,8],head_nums=[2,4,8,16],depths=[2,2,2,2],mlp_ratios=[4,4,2,2],act=act,window_size=8,dilate_rate=4):
        super().__init__()
        self.down_layers=nn.ModuleList()
        self.downresblocks=nn.ModuleList()
        self.downattblocks=nn.ModuleList()
        self.upattblocks=nn.ModuleList()
        self.up_layers=nn.ModuleList()
        self.upcatblocks=nn.ModuleList()

        self.embed_layers=nn.Conv2d(indim,gfdim*ch_mult[0],3,1,1,padding_mode="reflect")
        self.embed_background=nn.Conv2d(indim,gfdim*ch_mult[0],3,1,1,padding_mode="reflect")
        self.up_ch_mult=ch_mult[::-1]

        self.conv_tr=nn.Conv2d(gfdim*ch_mult[0],gfdim*ch_mult[0]*2,3,1,1,padding_mode="reflect")
        self.conv_out=nn.Conv2d(gfdim*ch_mult[0],outdim,3,1,1,padding_mode="reflect")

        self.down_resnum=2
        self.up_resnum=1
        self.gfdim=gfdim
        cur_res=resolution
        self.startatt=1

        self.resolutions=[resolution//(2**i) for i in range(len(ch_mult))]
        self.register_buffer("freqs_cis",Rotation_matrix_3D_complex(int(gfdim//2),window_size))
        self.register_buffer("w",OverlapDilatedWindowWeight(window_size=window_size,dilate_rate=dilate_rate,resolution=resolution))
 
        for i in range(len(ch_mult)-1):
            self.downattblocks.append(WDWABlocks(gfdim*ch_mult[i],head_num=head_nums[i],depth=depths[i],mlp_ratio=mlp_ratios[i],window_size=window_size,dilate_rate=dilate_rate))
            self.down_layers.append(patch_merge(gfdim*ch_mult[i],gfdim*ch_mult[i+1]))
            cur_res=cur_res//2 

            self.up_layers.append(patch_unmerge(gfdim*ch_mult[i+1],gfdim*ch_mult[i]))
            self.upcatblocks.append(catFusion(gfdim*ch_mult[i]*2,gfdim*ch_mult[i]))
            self.upattblocks.append(WDWABlocks(gfdim*ch_mult[i],head_num=head_nums[i],depth=depths[i],mlp_ratio=mlp_ratios[i],window_size=window_size,dilate_rate=dilate_rate))

        self.midatt= WDWABlocks(gfdim*ch_mult[-1],head_num=head_nums[-1],depth=depths[-1],mlp_ratio=mlp_ratios[-1],window_size=window_size,dilate_rate=dilate_rate)

    def forward(self,x):
        feature=[]
        b=self.embed_background(x)
        x=self.embed_layers(x)
        feature.append(x)
        k=len(self.up_ch_mult)-self.startatt
        for i in range(k):
            x=self.downattblocks[i](x,self.freqs_cis,self.w[:,:,:self.resolutions[i],:self.resolutions[i]])
            feature.append(x)
            x=self.down_layers[i](x)          
        x=self.midatt(x,self.freqs_cis,self.w[:,:,:self.resolutions[k],:self.resolutions[k]])
        feature=feature[::-1]

        for i in range(1,k+1):  
            x=self.up_layers[-i](x) 
            x=torch.cat([x,feature[i-1]],dim=1)
            x=self.upcatblocks[-i](x)
            x=self.upattblocks[-i](x,self.freqs_cis,self.w[:,:,:self.resolutions[-i-1],:self.resolutions[-i-1]])
        #PM-FS
        x = self.conv_tr(x)
        x=x[:,:self.gfdim]*b-x[:,self.gfdim:]+b
        x=self.conv_out(x)
        return x

