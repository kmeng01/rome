from util.hparams import HyperParams
from dataclasses import dataclass

@dataclass
class EFKHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    counterfact: bool
    zsre: bool
