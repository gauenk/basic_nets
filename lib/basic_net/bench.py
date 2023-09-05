
"""

Compare with deformable convolution

"""

# -- imports --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- summary --
from torchinfo import summary as th_summary
from functools import partial
from easydict import EasyDict as edict
from dev_basics import net_chunks
import copy
dcopy = copy.deepcopy

# -- model io --
import importlib

# -- data --
import data_hub

# -- optical flow --
# from dev_basics import flow

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.misc import optional
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


def run_img_bench(cfg,net=None):

    # -- init --
    th.cuda.init()
    timer = ExpTimer()
    memer = GpuMemer()

    # -- read data --
    in_chnls = optional(cfg,'in_chnls',3)
    nframes = optional(cfg,'bench_nframes',-1)
    H,W = [int(s) for s in optional(cfg,'bench_hw',"128_128").split("_")]
    if nframes > 0:
        img = th.zeros((32,nframes,in_chnls,H,W),device="cuda")
    else:
        img = th.zeros((32,in_chnls,H,W),device="cuda")

    # -- get net --
    if net is None:
        net = basic_net.load_model(cfg).to(device)

    # -- run search before for refine --
    out = net(img)

    # -- time forward --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                out = net(img)

    # -- enable gradient --
    img = img.requires_grad_(True)

    # -- time forward --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            out = net(img)

    # -- time bwd --
    loss = th.mean(out)
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # -- format results --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        results["res_%s"%key] = res
        results["alloc_%s"%key] = alloc

    return results
