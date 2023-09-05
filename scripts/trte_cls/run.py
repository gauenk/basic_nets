
import os
import datetime
import pandas as pd
from pathlib import Path
import cache_io
import torch as th
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from easydict import EasyDict as edict

# -- basic network --
import basic_net
from basic_net import run_img_bench
from basic_net.utils import save_network

# -- bench --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt


def get_data(name,batch_size_train,batch_size_test):

    DataClass = getattr(tv.datasets,name)
    root = "/home/gauenk/Documents/data/"
    train = th.utils.data.DataLoader(
        DataClass(root, train=True, download=True,
                                   transform=tv.transforms.Compose([
                                       tv.transforms.ToTensor(),
                                       tv.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True, pin_memory=True)
    test = th.utils.data.DataLoader(
        DataClass(root, train=False, download=True,
                                   transform=tv.transforms.Compose([
                                       tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
        batch_size=batch_size_test, shuffle=True, pin_memory=True)
    return train,test

def train(net,optimizer,train_loader,epoch,
          train_losses,train_counter,log_interval):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to("cuda")
        target = target.to("cuda")
        optimizer.zero_grad()
        output = net(data)
        # print(output,target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            pred = output.data.max(1, keepdim=True)[1]
            acc = pred.eq(target.data.view_as(pred)).sum()/len(output)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), acc.item()))

        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # th.save(network.state_dict(), '/results/model.pth')
      # th.save(optimizer.state_dict(), '/results/optimizer.pth')

def test(net,test_loader,test_losses,test_accs):
    net.eval()
    test_loss = 0
    correct = 0
    with th.no_grad():
        for data, target in test_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            output = net(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        test_accs.append((100. * correct / len(test_loader.dataset)).item())

def run(cfg):

    # -- init --
    n_epochs = cfg.nepochs
    batch_size_train = 64
    # batch_size_test = 1000
    batch_size_test = 1000
    # batch_size_test = 64
    momentum = 0.5
    log_interval = 10
    gamma = 0.7

    random_seed = 1
    th.backends.cudnn.enabled = False
    th.manual_seed(random_seed)
    device = "cuda:0"

    # -- read data --
    train_loader,test_loader = get_data(cfg.dname,batch_size_train,batch_size_test)

    # -- get net --
    net = basic_net.load_model(cfg).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # -- benchmark --
    results = run_img_bench(cfg,net)

    # -- init train results --
    train_losses = []
    train_counter = []
    test_losses = []
    test_accs = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


    # -- run over epochs --
    for epoch in range(1, n_epochs + 1):

        # -- train --
        train(net,optimizer,train_loader,epoch,
              train_losses,train_counter,log_interval)

        # -- test --
        test(net,test_loader,test_losses,test_accs)

        # -- update --
        scheduler.step()

    # -- save --
    path = "output/trte_cls/%s_%s/" % (cfg.dname,cfg.method_name)
    name = "net_%s.pth" % str(datetime.datetime)
    net_fn = save_network(net,path,name)
    results.test_accs = test_accs
    results.final_test = test_accs[-1]
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
    exp_data = {"cfg":cfg,
                "listed0":{"dname":["MNIST","CIFAR10","CIFAR100"],
                           "ncls":[10,10,100],"in_chnls":[1,3,3]},
                "listed1":{
                    "method_name":["conv","deform","stnls","stnls+offsets","mha"],
                    "nheads":[-1,-1,1,9,1],
                    "k":[-1,-1,9,1,-1],
                    "ws":[-1,-1,7,3,-1],
                    "stride1":[-1,-1,1.,1.,-1]}}
    exps = cache_io.exps.unpack(exp_data)

    # -- check exps --
    keys = ["dname","ncls","in_chnls","method_name","nheads","k","ws","stride1"]
    for exp in exps:
        print([exp[k] for k in keys])
    # exit()

    # -- run cached experiments --
    results = cache_io.run_exps(exps,run,name=".cache_io/trte_cls",
                                proj_name="basic_trte_cls",
                                skip_loop=False,records_fn="record_cls",
                                records_reload=True,clear=False,use_wandb=False)

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
    """
    main()
