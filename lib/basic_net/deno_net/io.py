
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
# from .img_net import BasicNet as BasicNetImg
# from .vid_net import BasicNet as BasicNetVid
from basic_net.modules.nls_deno import NlsConv2d as NlsDeno
from basic_net.modules.dcn_deno import DeformableConv2d as DcnDeno
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
    if cfg.method_name in ["dcn","deform"]:
        model = DcnDeno(cfg.nftrs,cfg.nftrs)
    elif cfg.method_name == "stnls":
        model = NlsDeno(cfg.nftrs,cfg.nftrs,cfg.k,cfg.ws,cfg.ps,
                        cfg.nheads,cfg.stride0,cfg.stride1)
    else:
        raise ValueError(f"Uknown basic deno type [{cfg.deno_type}]")

    # if cfg.deno_type == "img":
    #     model = BasicNetImg(cfg,cfg.method_name,cfg.in_chnls)
    # elif cfg.deno_type == "vid":
    #     model = BasicNetImg(cfg,cfg.method_name,cfg.in_chnls)
    # else:
    #     raise ValueError(f"Uknown basic deno type [{cfg.deno_type}]")

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
    pairs = {"method_name":"conv","nftrs":32,"deno_type":"img"}
    return pairs

