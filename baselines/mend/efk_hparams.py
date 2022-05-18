from util.hparams import HyperParams


class EFKHyperParams(HyperParams):
    KEYS = [
        "lr_scale",
        "n_toks",
        "model_name",
        "counterfact",
        "zsre"
    ]
