from util.hparams import HyperParams
from dataclasses import dataclass


@dataclass
class MENDHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    mini: bool
    zsre: bool
