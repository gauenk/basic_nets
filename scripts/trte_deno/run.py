

# -- basic --
import os
import datetime
import pandas as pd
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
import numpy as np


# -- deep learning --
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# -- data aug --
import torchvision as tv
import torch.nn.functional as F

# -- caching --
import cache_io

# -- data --
import data_hub
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils.misc import set_seed
from basic_net.utils.mask import bbox2mask,random_bbox


# -- basic network --
import basic_net
from basic_net import run_img_bench
from basic_net.utils import save_network
from basic_net.modules import VideoDenoWrap

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


def get_noisy(cfg,clean):
    if cfg.ntype == "g":
        noisy = clean + th.randn_like(clean)*cfg.sigma/255.
    # elif cfg.ntype == "infill":
    #     H,W = clean.shape[-2:]
    #     ishape = (H,W)
    #     mshape = (H//4,W//4)
    #     delta = 5
    #     margin = 5
    #     mask = bbox2mask(clean.shape[-2:], random_bbox(ishape,mshape,delta,margin))[...,0]
    #     mask = th.from_numpy(mask).to(clean.device).view(1,1,H,W)
    #     noisy = clean*(1. - mask) + mask*th.randn_like(clean)
    #     # print(th.mean((noisy - clean)**2))
    else:
        pass
    return noisy

def train(cfg,net,optimizer,train_loader,epoch,
          train_losses,train_counter,log_interval):
    net.train()
    for batch_idx, sample in enumerate(train_loader):
        clean = sample['clean']/255.
        clean = clean.to("cuda")
        clean -= clean.min()
        clean /= clean.max()
        noisy = get_noisy(cfg,clean)
        optimizer.zero_grad()
        deno = net(noisy)
        loss = F.mse_loss(deno, clean)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            psnrs = np.mean(compute_psnrs(deno,clean,1.))
            base_psnrs = np.mean(compute_psnrs(noisy,clean,1.))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tNoisy: {:.2f}\tDeno: {:.2f}'.format(
                epoch, batch_idx * len(clean), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                base_psnrs.item(), psnrs.item()))

        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # th.save(network.state_dict(), '/results/model.pth')
      # th.save(optimizer.state_dict(), '/results/optimizer.pth')

def test(cfg,net,test_loader,test_losses,test_psnrs):
    net.eval()
    test_loss = 0
    psnr = 0
    n = 0
    with th.no_grad():
        for sample in test_loader:
            clean = sample['clean']/255.
            clean = clean.to("cuda")
            clean -= clean.min()
            clean /= clean.max()
            noisy = get_noisy(cfg,clean)
            deno = net(noisy)
            test_loss += F.mse_loss(deno, clean).item()
            psnr += np.mean(compute_psnrs(deno,clean,1.))
            n += 1
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        psnr = psnr / n
        print('\nTest set: Avg. loss: {:.4e}, Deno: {:.2f}\n'.format(test_loss, psnr))
        test_psnrs.append(psnr)

def run(cfg):

    # -- init --
    n_epochs = cfg.nepochs
    momentum = 0.5
    log_interval = 10
    gamma = 0.7

    random_seed = 1
    th.backends.cudnn.enabled = False
    th.manual_seed(random_seed)
    device = "cuda:0"

    # -- read data --
    data,loaders = data_hub.sets.load(cfg)
    train_loader = loaders.tr
    test_loader =  loaders.te

    # -- get net --
    net_module = basic_net.load_model(cfg).to(device)
    net = VideoDenoWrap(net_module,cfg.spynet_path,cfg.noffsets).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # -- benchmark --
    results = run_img_bench(cfg,net)

    # -- init train results --
    train_losses = []
    train_counter = []
    test_losses = []
    test_psnrs = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # -- run over epochs --
    for epoch in range(1, n_epochs + 1):

        # -- train --
        train(cfg,net,optimizer,train_loader,epoch,
              train_losses,train_counter,log_interval)

        # -- test --
        test(cfg,net,test_loader,test_losses,test_psnrs)

        # -- update --
        scheduler.step()

    # -- test ood --
    _cfg = dcopy(cfg)
    _cfg.sigma = cfg.sigma_ood
    test(_cfg,net,test_loader,test_losses,test_psnrs)

    # -- save --
    path = "output/trte_cls/%s_%s/" % (cfg.dname,cfg.method_name)
    name = "net_%s.pth" % str(datetime.datetime)
    net_fn = save_network(net,path,name)
    results.test_psnrs = test_psnrs
    results.test_ood = test_psnrs[-1]
    results.final_test = test_psnrs[-2]
    results.save_fn = net_fn

    return results

def main():


    # -- init exp --
    pid = os.getpid()
    print("PID: ",pid)

    # -- experiment grid --
    cfg = edict()
    cfg.seed = 123
    cfg.lr = 1e-3
    cfg.stride0 = 1
    cfg.anchor_self = False
    cfg.ps = 1
    cfg.nepochs = 15
    cfg.itype_fwd = "float"
    cfg.itype_bwd = "float"
    cfg.task = "deno"
    cfg.ntype = "g"
    cfg.spynet_path = "./weights/spynet_sintel_final-3d2a1287.pth"
    cfg.batch_size_tr = 5
    cfg.batch_size_te = 1
    cfg.nframes = 5
    cfg.bench_nframes = 5
    cfg.nsamples_tr = 100
    exp_data = {"cfg":cfg,
                "listed2":{
                    "method_name":["deform","stnls"],
                    "nheads":[1,3],
                    "k":[9,3],
                    "noffsets":[9,9],
                    "ws":[7,3],
                    "stride1":[1.,1.]},
                "listed0":{"sigma":[30,50,10]},
                "listed1":{"dname":["davis"],
                           "isize":["256_256"],
                           "nframes_tr":[5]}}
    exps = cache_io.exps.unpack(exp_data)

    # -- check exps --
    keys = ['sigma',"method_name","nheads","k","ws","stride1"]
    for exp in exps:
        print([exp[k] for k in keys])
    # exit()
    exps = [exps[0]]

    # -- run cached experiments --
    results = cache_io.run_exps(exps,run,name=".cache_io/trte_deno",
                                proj_name="basic_trte_deno",
                                skip_loop=False,records_fn="record_cls",
                                records_reload=False,clear=False,use_wandb=False)

    # -- viz --
    vals = ["timer_fwd","timer_bwd","final_test"]
    pivot = pd.pivot_table(results,values=["final_test"],index=["method_name"],
                           columns=["dname"],aggfunc=["mean"])
    print(pivot)
    keys = ["method_name","timer_fwd","timer_bwd"]
    res = results[keys].groupby("method_name").agg("mean")
    print(res)

if __name__ == "__main__":
    """

    -=-=-=- inpaint -=-=-=-=-

    """
    main()
