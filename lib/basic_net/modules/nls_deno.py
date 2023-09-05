import torch as th
import torchvision.ops
from torch import nn
from easydict import EasyDict as edict
import stnls
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from einops import rearrange

class NlsConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,
                 k,ws,ps=1,nheads=1,
                 stride0=1,stride1=1.,
                 itype_fwd="float",itype_bwd="float"):
        super(NlsConv2d, self).__init__()

        # -- init nls --
        cfg = edict()
        cfg.ps = ps
        cfg.k = k
        cfg.ws = ws
        cfg.wt = 0
        cfg.stride0 = stride0
        cfg.stride1 = stride1
        # cfg.search_name = "nls"
        cfg.full_ws = False
        cfg.search_name = "paired"
        cfg.dist_type = "prod"
        cfg.itype_fwd = itype_fwd
        cfg.itype_bwd = itype_bwd
        cfg.nheads = nheads
        self.nheads = nheads
        self.dist_type = cfg.dist_type
        self.search = stnls.search.init(cfg)
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        KHD = cfg.k * cfg.nheads
        k = cfg.ws*cfg.ws if k == -1 else k

        self.offset_conv = nn.Conv2d(in_channels*2+2,2*nheads,
                                     kernel_size=3,stride=1,
                                     padding=1,dilation=1,bias=True)


        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels*2+2,1 * k * nheads,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        dilation=1,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      dilation=1,
                                      bias=True)

    def run_search(self,frame_i,frame_j,s_offsets):

        def get_grid(H,W,dtype,device):
            grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                            th.arange(0, W, dtype=dtype, device=device))
            grid = th.stack((grid_y, grid_x), 2).float()  # W(x), H(y), 2
            grid = rearrange(grid,'H W two -> two H W')
            return grid

        # -- offsets from conv --
        B,C,H,W = frame_i.shape
        s_offsets = s_offsets.view(B,-1,2,H,W)
        dists,inds = self.search(frame_i[:,None].contiguous(),
                                 frame_j[:,None].contiguous(),s_offsets)
        inds = th.cat([th.zeros_like(inds[...,[0]]),inds],-1)
        assert not(th.any(th.isinf(dists)).item())

        # -- inds to offsets --
        grid = get_grid(H,W,frame_i.dtype,frame_i.device)
        inds = inds[...,-2:]
        inds_n = rearrange(inds,'b HD (H W) k two -> b (HD k) two H W',H=H,W=W)
        offset = inds_n - grid[None,None,]
        offset = offset.flip(2)
        offset = offset.reshape(B,-1,H,W) # flip(2)?

        return offset


    def forward(self, frame_i, frame_j, flow_ij):

        # -- compute offsets --
        B,C,H,W = frame_i.shape
        offset_input = th.cat([frame_i,frame_j,flow_ij],-3)
        offset = self.offset_conv(offset_input)

        # -- run search --
        offset = self.run_search(frame_i,frame_j,offset)

        # -- aggregate with deform conv2d --
        modulator = 2. * th.sigmoid(self.modulator_conv(offset_input))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=frame_j,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)


        return x
