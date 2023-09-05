"""

Wrap a core module for simple video denoising

"""

import torch as th
import torchvision.ops
from torch import nn
from easydict import EasyDict as edict
import stnls
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from einops import rearrange
from einops.layers.torch import Rearrange
from .spynet import SpyNet


class VideoDenoWrap(nn.Module):
    def __init__(self,net,spynet_path,noffsets,in_channels=3):
        super(VideoDenoWrap, self).__init__()

        # -- init network --
        self.paired_net = net
        self.wt = 1
        self.st = 2*self.wt+1

        # -- offsets across time --
        self.spynet = SpyNet(spynet_path)

        # -- self offsets --
        self.offset_conv = nn.Conv2d(in_channels, 2,
                                     kernel_size=3,stride=1,
                                     padding=1,dilation=1,bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # -- feature extraction --
        self.init_ftrs = nn.Sequential(*[
            Rearrange('n d c h w -> n c d h w'),
            nn.Conv3d(3,32,kernel_size=(1,3,3),padding=1),
            nn.ReLU(),
            nn.Conv3d(32,32,kernel_size=(1,3,3),padding=1),
            nn.ReLU(),
            Rearrange('n c d h w -> n d c h w')
        ])
        self.conv = nn.Sequential(*[nn.Conv2d(32*self.st,32,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32,32,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32,3,kernel_size=3,padding=1)
        ])

    def get_adj_frames(self,ti,wt,T):

        # -- init --
        adj = []

        # -- compute offsets --
        swap = False
        t_inc = 0
        prev_t = ti
        t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
        t_max = min(T-1,ti + wt - t_shift);
        tj = ti

        # -- run --
        for _tj in range(2*wt+1):
            prev_t = tj
            tj = prev_t + t_inc
            swap = tj > t_max
            t_inc = 1 if (t_inc == 0) else t_inc
            t_inc = -1 if swap else t_inc
            tj = ti-1 if swap else tj
            prev_t = ti if swap else prev_t
            adj.append(tj)
        return adj

    # def run_flows(self,vid):
    #     # -- compute first order indices --
    #     fflow,bflow = self.spynet.run_pairs(vid)

    #     acc_flows = stnls.nn.accumulate_flow(fflow_gt,bflow_gt)
    #     fflow,bflow = acc_flows.fflow,acc_flows.bflow

    #     sflow = []
    #     for ti in range(T):
    #         sflow.append(self.offset_conv(vid[ti]))
    #     sflow = th.stack(sflow)
    #     return fflow,bflow,sflow

    # def get_flow(self,ti,tj,fflow,bflow,sflow):
    #     if ti == tj:
    #         return sflow[ti]
    #     if ti > tj:
    #         return fflow[ti]
    #     elif ti < tj:
    #         return bflow[tj]

    def forward(self,vid):

        # -- init extract features --
        ftrs = self.init_ftrs(vid)
        # print("vid.shape: ",vid.shape)
        # print("ftrs.shape: ",ftrs.shape)

        # -- run pairwise method --
        T = vid.shape[1]
        deno = []
        for ti in range(T):
            ftrs_i = []
            for tj in self.get_adj_frames(ti,self.wt,T):
                if ti != tj:
                    flow = self.spynet(vid[:,ti],vid[:,tj])
                else:
                    flow = self.offset_conv(vid[:,ti])
                ftrs_i.append(self.paired_net(ftrs[:,ti],ftrs[:,tj],flow))
            ftrs_i = th.cat(ftrs_i,1)# cat along channels
            deno_i = self.conv(ftrs_i)
            deno.append(deno_i)
        deno = th.stack(deno,1)

        return deno
