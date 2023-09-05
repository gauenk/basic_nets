
import torch as th
from torch import nn
from basic_net.modules.dcn import DeformableConv2d
from basic_net.modules.nls_deno import NlsConv2d
from basic_net.modules.mha import WrapMultiHeadAttn

def init_conv(cfg,name):
    if name == "deform":
        return DeformableConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
    elif name == "conv":
        return nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
    elif name == "stnls":
        return NlsConv2d(32,32,cfg.k,cfg.ws,ps=cfg.ps,nheads=cfg.nheads,
                         stride0=cfg.stride0,stride1=cfg.stride1,
                         itype_fwd=cfg.itype_fwd,itype_bwd=cfg.itype_bwd)
    elif name == "mha":
        return WrapMultiHeadAttn(32,cfg.nheads)
    else:
        raise ValueError("Uknown.")

class BasicNet(nn.Module):
    def __init__(self,cfg,method_name,in_chnls):

        super(BasicNet, self).__init__()

        self.conv1 = nn.Conv2d(in_chnls, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = init_conv(cfg,method_name)
        self.conv4 = init_conv(cfg,method_name)
        self.conv5 = nn.Conv2d(32, in_chnls, kernel_size=3, stride=1, padding=1, bias=True)


        # self.pool = nn.MaxPool2d(2)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(32, ncls)

    def forward(self, x):
        x = th.relu(self.conv1(x))
        # x = self.pool(x) # [14, 14]
        x = th.relu(self.conv2(x))
        x = th.relu(self.conv3(x))
        # x = self.pool(x) # [7, 7]
        x = th.relu(self.conv4(x))
        x = th.relu(self.conv5(x))
        # x = self.gap(x)
        # x = x.flatten(start_dim=1)
        # x = self.fc(x)
        return x
