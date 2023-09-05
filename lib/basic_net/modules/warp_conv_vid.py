import torch
import torchvision.ops
from torch import nn

from easydict import EasyDict as edict
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from einops import rearrange


class WarpConvVid(nn.Module):
    def __init__(self,chnls,nheads=1):
        super(WarpConvVid, self).__init__()

        stride = 1
        kernel_size = 1
        self.padding = 0
        self.dilation = 1
        self.stride = stride
        self.proj_q = nn.Conv2d(chnls,
                                chnls,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                bias=True)
        self.proj_k = nn.Conv2d(chnls,chnls,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                bias=True)
        kernel_size = 3
        padding = 1
        self.proj_v = nn.Conv2d(chnls,chnls,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=self.dilation,
                                bias=True)
        self.attn = nn.MultiheadAttention(32,nheads,batch_first=True)


    def forward(self, x):
        B,C,H,W = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        #v = x
        v = self.proj_v(x)
        q = rearrange(q,'b c h w -> b (h w) c')
        k = rearrange(k,'b c h w -> b (h w) c')
        v = rearrange(v,'b c h w -> b (h w) c')
        x,_ = self.attn(q,k,v)
        x = rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        return x

