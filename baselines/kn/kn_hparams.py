from dataclasses import dataclass

from util.hparams import HyperParams


@dataclass
class KNHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    refine: bool
    batch_size: int
    steps: int
    adaptive_threshold: float
    p: float
