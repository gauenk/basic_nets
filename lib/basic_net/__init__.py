
# -- code api --
from . import cls_net
from . import deno_net
from . import lightning
from .cls_net import extract_config as extract_config_cls
from .cls_net import extract_config as extract_model_config_cls

# -- benchmark --
from .bench import run_img_bench

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'task','cls')
    if mtype in ["cls"]:
        return cls_net.load_model(cfg)
    elif mtype in ["deno"]:
        return deno_net.load_model(cfg)
    else:
        raise ValueError(f"Uknown model type [{mtype}]")
