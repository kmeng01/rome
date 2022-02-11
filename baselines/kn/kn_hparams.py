from util.hparams import HyperParams


class KNHyperParams(HyperParams):
    KEYS = [
        "lr_scale",
        "n_toks",
        "model_name",
        "refine",
        "batch_size",
        "steps",
        "adaptive_threshold",
        "p",
    ]
