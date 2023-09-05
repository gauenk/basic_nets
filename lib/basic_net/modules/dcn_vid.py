import torch
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def paired_vids(self, vid0, vid1, acc_flows, wt):
        dists,inds = [],[]
        T = vid0.shape[1]
        zflow = th.zeros_like(acc_flows.fflow[:,0,0])
        for ti in range(T):
            # if ti != 1: continue

            swap = False
            t_inc = 0
            prev_t = ti
            t_shift = min(0,ti-wt) + max(0,ti + wt - (T-1))
            t_max = min(T-1,ti + wt - t_shift);
            # print(t_shift,t_max)
            tj = ti

            dists_i,inds_i = [],[]
            for _tj in range(2*wt+1):

                # -- update search frame --
                prev_t = tj
                tj = prev_t + t_inc
                swap = tj > t_max
                t_inc = 1 if (t_inc == 0) else t_inc
                t_inc = -1 if swap else t_inc
                tj = ti-1 if swap else tj
                prev_t = ti if swap else prev_t
                # print(ti,tj,t_inc,swap)

                frame0 = vid0[:,ti]
                frame1 = vid1[:,tj]
                if ti == tj:
                    flow = zflow
                elif ti < tj:
                    # print("fwd: ",ti,tj,tj-ti-1)
                    # flow = acc_flows.fflow[:,tj - ti - 1]
                    flow = acc_flows.fflow[:,ti,tj-ti-1]
                elif ti > tj:
                    # print("bwd: ",ti,tj,ti-tj-1)
                    # flow = acc_flows.bflow[:,ti - tj - 1]
                    flow = acc_flows.bflow[:,ti,ti-tj-1]
                flow = flow.float()
                dists_ij,inds_ij = self.forward(frame0,frame1,flow)
                inds_t = tj*th.ones_like(inds_ij[...,[0]])
                inds_ij = th.cat([inds_t,inds_ij],-1)
                dists_i.append(dists_ij)
                inds_i.append(inds_ij)
            dists_i = th.cat(dists_i,-1)
            inds_i = th.cat(inds_i,-2)
            dists.append(dists_i)
            inds.append(inds_i)
        dists = th.cat(dists,-2)
        inds = th.cat(inds,-3)
        return dists,inds


    def run_deform_attn(self,q,kv):
        v = deform_attn(proj_q, kv, offset, self.kernel_h, self.kernel_w, self.stride,
                        self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups,
                        self.clip_size).view(b, t, self.proj_channels, h, w)

    def forward(self, x, flows):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
