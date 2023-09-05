
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- searching --
import stnls

# -- network --
from . import menu
from .net import BasicNet
from .scaling import Downsample,Upsample # defaults

# -- search/normalize/aggregate --
import stnls

# -- io --
from dev_basics import arch_io

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction


# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #        Config
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- init --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda")

    # -- unpack local vars --
    local_pairs = {"io":io_pairs(),
                   "arch":arch_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg
    dep_pairs = {"search":stnls.search.extract_config}
    cfgs = dcat(cfgs,econfig.extract_dict_of_econfigs(cfg,dep_pairs))
    cfg = dcat(cfg,econfig.flatten(cfgs))

    # -- init model --
    model = BasicNet(cfg,cfg.method_name,cfg.ncls,cfg.in_chnls)

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    # model = model.to(device)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs

def arch_pairs():
    pairs = {"method_name":"conv","ncls":10,"in_chnls":1}
    return pairs

