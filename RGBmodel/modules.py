import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
def act(inplace=True):
    return nn.SiLU(inplace=inplace)

class RLN(nn.Module):
	r"""Revised LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias
     
def Normalize(in_channels, num_groups=16,eps=1e-6):
    return RLN(in_channels)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=act(inplace=True), drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features,1,bias=False)
        self.act = act_layer
        self.fc2 = nn.Conv2d(hidden_features, in_features,1,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def Rotation_matrix_3D_complex(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 3)[: (dim // 3)].float() / dim)).to("cuda")#[1,1/10000**0.01,2...,dim//3]
    t = torch.arange(end, device=freqs.device) # [0,1,2...end-1]
    freqs_cis = torch.outer(t, freqs).float()#end,dim//3
    freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)#end,dim//3,cosj+sinj
    return freqs_cis#L dim//3

def HS_PE_complex(x: torch.Tensor,freqs_cis: torch.Tensor,):
    b,c,h,w=x.shape
    x = rearrange(x,"b (c z) h w -> b w h c z",z=3).contiguous()#b,w,h,dim//3,3
    x[...,0:2] = torch.view_as_real(torch.view_as_complex(x[...,0:2].contiguous())*freqs_cis[:h])
    
    x = rearrange(x,"b w h c z -> b h w c z",w=w).contiguous()#b,w*dim//3,h,3
    x[...,1:3] = torch.view_as_real(torch.view_as_complex(x[...,1:3].contiguous())*freqs_cis[:w])
    return rearrange(x,"b h w c z -> b (c z) h w",h=h).contiguous()

class MHSA(nn.Module):#b h w c
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,q,k,v):
        dk = torch.tensor(k.shape[-1])#b,c,hw
        matmul_qk = (q@k.transpose(-1,-2))  # (..., seq_len_q, seq_len_k)
        scaled_attention_logits = self.softmax(matmul_qk/torch.sqrt(dk))#+p
        v = scaled_attention_logits@v  # (..., seq_len_q)x
        
        return v
import math
def OverlapDilatedWindowPartion(x, window_size,dilate_rate):#ODWP，x:(b,c,h,w)
    Wsize=window_size+dilate_rate
    stride=Wsize//2
    numh=math.ceil(x.shape[-2]/Wsize)
    numw=math.ceil(x.shape[-1]/Wsize)
    x=F.pad(x, (0,numw*Wsize-x.shape[-1],0,numh*Wsize-x.shape[-2]), mode='reflect')#右、下填充
    x=torch.cat([x,F.pad(x[:,:,stride:,:], (0,0,0,stride), mode='reflect')],axis=0)#[原图，向下滑动]
    x=torch.cat([x,F.pad(x[:,:,:,stride:], (0,stride,0,0), mode='reflect')],axis=0)#[原图，向下滑动,原图向右滑动，原图向下向右滑动]
    windows = rearrange(x,"gb c (nh h) (nw w) -> (nh nw gb) c h w",h=Wsize,w=Wsize)
    return windows[:,:,:window_size,:window_size]

def OverlapDilatedWindowReverse(windows,window_size,dilate_rate,H, W):#ODWR
    Wsize=window_size+dilate_rate
    stride=Wsize//2
    numh=math.ceil(H/Wsize)
    numw=math.ceil(W/Wsize)
    bH=numh*Wsize
    bW=numw*Wsize
    windows = F.pad(windows, (0,dilate_rate,0,dilate_rate), mode='constant',value=0)
    x=rearrange(windows,"(nh nw g b) c h w -> g b c (nh h) (nw w)",g=4,nh=numh,nw=numw)

    x[1]=F.pad(x[1], (0,0,stride,0), mode='constant',value=0)[:,:,:bH,:bW]#上填充
    x[2]=F.pad(x[2], (stride,0,0,0), mode='constant',value=0)[:,:,:bH,:bW]#左填充
    x[3]=F.pad(x[3], (stride,0,stride,0), mode='constant',value=0)[:,:,:bH,:bW]#左填充、上填充
    return torch.sum(x,axis=0)[:,:,:H,:W]

class WeightedDilatedWindowAttention(nn.Module):#WDWA
    def __init__(self,*,indim,outdim=None,head_num=8,window_size=8,dilate_rate=4):
        super().__init__()
        self.head_num=head_num
        outdim=indim or outdim
        self.window_size=window_size
        self.dilate_rate=dilate_rate
        self.stride=(window_size+dilate_rate)//2
        self.Wsize=window_size+dilate_rate
        self.att=MHSA()
       
    def forward(self,x,freqs_cis,weight):
        B,C,H,W=x.shape#b,(3 nhead dim),h,w 
        x=OverlapDilatedWindowPartion(x,self.window_size,self.dilate_rate)#window_nums*window_nums*4*b (3 nhead dim) window_size window_size
        qkv = rearrange(x,"nhnwgb (k n d) h w -> k (nhnwgb n) d h w",k=3,n=self.head_num)
        qkv[0],qkv[1]=HS_PE_complex(qkv[0],freqs_cis),HS_PE_complex(qkv[1],freqs_cis)
        qkv=rearrange(qkv,"k nhnwgbn d h w -> k nhnwgbn (h w) d")
        
        att=self.att(qkv[0],qkv[1],qkv[2])#window_nums*window_nums*4*b*num_head window_size*window_size head_dim

        att=rearrange(att,"(nhnwgb n) (h w) d -> nhnwgb (n d) h w",n=self.head_num,h=self.window_size,w=self.window_size)
        att=OverlapDilatedWindowReverse(att,self.window_size,self.dilate_rate,H,W)#b*4,c,h,w
 
        return att/weight

class WDWAFormer(nn.Module):
    def __init__(self,*,indim,outdim=None,head_num,norm_layer=Normalize,window_size=8,dilate_rate=4,mlp_ratio=2,qkv_bias=False,weight=None):
        super().__init__()
        self.head_num=head_num
        self.outdim=indim or outdim
        self.norm1=norm_layer(indim)
        self.normout=norm_layer(self.outdim)
        self.qkv=nn.Conv2d(self.outdim,indim*3,1,bias=qkv_bias)
        self.DWC=nn.Conv2d(indim,indim,3,1,1,padding_mode="reflect",groups=indim)
        self.att=WeightedDilatedWindowAttention(indim=indim,head_num=head_num,window_size=window_size,dilate_rate=dilate_rate)
        
        self.proj=nn.Conv2d(indim,self.outdim,1)
        self.mlp=Mlp(self.outdim,self.outdim*mlp_ratio)
        
    def forward(self,x,freqs_cis,weight):
        shortcut=x
        x, rescale, rebias = self.norm1(x)
        x=self.qkv(x)
        df=self.DWC(x[:,self.outdim*2:])
        x=self.att(x,freqs_cis,weight)
        x=self.proj(x+df)
        x = x * rescale + rebias
        x=x+shortcut
        
        shortcut =x
        x, rescale, rebias = self.normout(x)
        x=self.mlp(x)
        x = x * rescale + rebias
        x = x+shortcut
        return x

class SpFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SpFusion, self).__init__()
		
		self.height = height
		d = max(int(dim/reduction), 4)
		
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim, d, 1, bias=False), 
			nn.ReLU(),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)
		
		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape
		
		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)
		
		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(self.avg_pool(feats_sum))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out
class catFusion(nn.Module):
    def __init__(self,indim,outdim=None):
        super().__init__() 
        outdim= indim//2 if outdim is None else outdim
        self.fuse = nn.Conv2d(indim,outdim,3,1,1,padding_mode="reflect")
    def forward(self,x):
        x=self.fuse(x)
        return x
class Fusion(nn.Module):
    def __init__(self,indim,act=nn.ReLU(inplace=True)):
        super().__init__()
        self.sk=SpFusion(indim)       
        self.fuse = nn.Conv2d(indim*2,indim,3,1,1,bias=False)
    def forward(self,x,y):
        shortcut=self.sk([x,y])
        x=self.fuse(torch.cat([x,y],dim=1))
        return x+shortcut
    
def OverlapDilatedWindowWeight(window_size=10,dilate_rate=6,resolution=256):
    window_nums=math.ceil(resolution/(window_size+dilate_rate))
    w=torch.ones([window_nums*window_nums*4,1,window_size,window_size])
    w=OverlapDilatedWindowReverse(w,window_size,dilate_rate,resolution,resolution)#b*4,1,h,w
    return w

class WDWABlocks(nn.Module):
    def __init__(self,indim,head_num,depth=2,mlp_ratio=2,window_size=10,dilate_rate=6,qkv_bias=False):
        super().__init__()
        self.encodelayers=nn.ModuleList([WDWAFormer(indim=indim,head_num=head_num,window_size=window_size,dilate_rate=dilate_rate,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias) for l in range(depth)])

    def forward(self,x,freqs_cis,weight):
        for layer in self.encodelayers:
            x=layer(x,freqs_cis,weight)
        return x
class selfTransBlocks(nn.Module):
    def __init__(self,indim,head_num,depth=2,mlp_ratio=2,freqs_cis=None,window_size=8,qkv_bias=False):
        super().__init__()
        self.encodelayers=nn.ModuleList([selfTransBlock(indim=indim,head_num=head_num,qkv_bias=qkv_bias,freqs_cis=freqs_cis) for l in range(depth)])
    def forward(self,x):
        for layer in self.encodelayers:
            x=layer(x)
        return x

