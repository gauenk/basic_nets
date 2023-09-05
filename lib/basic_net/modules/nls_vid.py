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
        cfg.search_name = "nls"
        cfg.dist_type = "prod"
        cfg.itype_fwd = itype_fwd
        cfg.itype_bwd = itype_bwd
        cfg.nheads = nheads
        self.dist_type = cfg.dist_type
        self.search = stnls.search.init(cfg)
        self.stacking = stnls.tile.NonLocalStack(ps=cfg.ps,stride0=cfg.stride0,
                                                 itype_fwd="float",itype_bwd="float")
        KHD = cfg.k * cfg.nheads

        # -- conv qkv --
        ksize = 1
        self.proj_q = nn.Conv2d(in_channels=in_channels,
                                out_channels=32,
                                kernel_size=ksize,stride=1,
                                padding=0,dilation=1,bias=True)
        self.proj_k = nn.Conv2d(in_channels=in_channels,
                              out_channels=32,
                              kernel_size=ksize,stride=1,
                              padding=0,dilation=1,bias=True)
        # self.proj_v = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=32,
        #                       kernel_size=3,stride=1,
        #                       padding=1,dilation=1,bias=True)

        # nn.init.constant_(self.proj_q.weight, 1.)
        # nn.init.constant_(self.proj_q.bias, 0.)
        # nn.init.constant_(self.proj_k.weight, 1.)
        # nn.init.constant_(self.proj_k.bias, 0.)

        # self.proj_v = nn.Conv2d(in_channels=in_channels,
        #                         out_channels=out_channels,
        #                         kernel_size=ksize,stride=1,
        #                         padding=0,dilation=1,bias=True)
        k = cfg.ws*cfg.ws if k == -1 else k
        # self.proj_weights = nn.Linear(k,k,bias=True)

        # -- init conv --
        # ksize = (k,1,1)
        # pad = (0,0,0)
        stride = (k,1,1)
        ksize = (k,3,3)
        pad = (0,1,1)
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=ksize,stride=stride,
                              padding=pad,dilation=1,bias=True)


        # self.dev = nn.Conv2d(in_channels=in_channels,
        #                       out_channels=32,
        #                       kernel_size=3,stride=1,
        #                       padding=1,dilation=1,bias=True)

        # -- reweight --
        # self.weights = nn.Linear(k,k)


        #
        # -- offset --
        #

        stride = 1
        bias = True
        kernel_size = [3,3]
        self.padding = 1
        self.dilation = 1
        self.stride = stride
        M = KHD // 9
        # self.modulator_conv = nn.Conv2d(in_channels,
        #                                 M * kernel_size[0] * kernel_size[1],
        #                                 kernel_size=kernel_size,
        #                                 stride=stride,
        #                                 padding=self.padding,
        #                                 dilation=self.dilation,
        #                                 bias=True)

        # nn.init.constant_(self.modulator_conv.weight, 0.)
        # nn.init.constant_(self.modulator_conv.bias, 0.)

        # self.regular_conv = nn.Conv2d(in_channels=in_channels,
        #                               out_channels=out_channels,
        #                               kernel_size=kernel_size,
        #                               stride=stride,
        #                               padding=self.padding,
        #                               dilation=self.dilation,
        #                               bias=bias)

    def run_proj(self,x,name):
        proj = getattr(self,name)
        B = x.shape[0]
        x = rearrange(x,'b t c h w -> (b t) c h w')
        x = proj(x)
        x = rearrange(x,'(b t) c h w -> b t c h w',b=B)
        # # C = x.shape[1]
        # L = 3
        # get_pos = PositionalEncoding2D(L)
        # x = rearrange(x,'b c h w -> b h w c')
        # pos = get_pos(x[...,-L:])
        # # # print(x.shape,pos.shape)
        # x[...,-L:] = x[...,-L:] + pos
        # x = rearrange(x,'b h w c -> b c h w')

        return x

    def forward(self, x, flows):

        def get_grid(H,W,dtype,device):
            grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                            th.arange(0, W, dtype=dtype, device=device))
            grid = th.stack((grid_y, grid_x), 2).float()  # W(x), H(y), 2
            grid = rearrange(grid,'H W two -> two H W')
            return grid

        # -- search --
        B,C,H,W = x.shape
        # print(" -- x -- ")
        # print(x)
        # print("-="*20)
        # print("-="*20)

        # -- proj/positional embeds --
        q = self.run_proj(x,"proj_q")
        k = self.run_proj(x,"proj_k")

        # -- search --
        zflow = th.zeros((B,1,2,H,W),device=x.device,dtype=x.dtype)
        dists,inds = self.search(q,k,flows.fflow,flows.bflow)
        # dists,inds = self.search(q,q,zflow,zflow)
        # dists,inds = self.search(x[:,None],x[:,None],zflow,zflow)
        # inds = inds.detach()
        # print(dists)
        assert not(th.any(th.isinf(dists)).item())

        # # -- inds to offsets --
        # grid = get_grid(H,W,x.dtype,x.device)
        # inds = inds[...,1:]
        # inds_n = rearrange(inds,'b HD (H W) k two -> b (HD k) two H W',H=H,W=W)
        # print(inds_n[0][0])
        # offset = inds_n - grid[None,None,]
        # offset = offset.reshape(B,-1,H,W)

        # # -- use in deform conv2d --
        # modulator = 2. * th.sigmoid(self.modulator_conv(x))
        # # op = (n - (k * d - 1) + 2p / s)
        # print(self.regular_conv.weight.shape,dists.shape)
        # x = torchvision.ops.deform_conv2d(input=x,
        #                                   offset=offset,
        #                                   weight=self.regular_conv.weight*dists,
        #                                   bias=self.regular_conv.bias,
        #                                   padding=self.padding,
        #                                   mask=modulator,
        #                                   stride=self.stride,
        #                                   dilation=self.dilation)


        #
        #
        #

        # -- reweight --
        if self.dist_type == "l2":
            weights = nn.functional.softmax(-10*dists,-1)
        else:
            weights = nn.functional.softmax(dists/10.,-1)
        # weights = self.proj_weights(weights)
        # weights = self.proj_weights(th.exp(-10*dists))

        # -- aggregate --
        # v = self.proj_v(x)[:,None]
        # x = self.stacking(v,weights,inds)
        # inds = th.cat([th.zeros_like(inds[...,[0]]),inds],-1)
        # print(inds.shape)
        x = self.stacking(x[:,None],weights,inds)
        x = rearrange(x,'b hd k 1 c h w -> b (hd c) k h w')
        # print(" --- stacking --- ")
        # print(x)

        # print("x.shape: ",x.shape)
        # # weights = self.proj_weights(weights)
        # x = th.mean(x,2)

        # # -- transform --
        # # weights = self.proj_weights(weights)
        # # weights
        # print(self.regular_conv.weight.shape,self.conv.weight.shape)
        x = self.conv(x)[:,:,0]


        return x
