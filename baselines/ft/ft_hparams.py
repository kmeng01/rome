from util.hparams import HyperParams


class FTHyperParams(HyperParams):
    KEYS = [
        # Method
        "layers",
        "num_steps",
        "lr",
        "weight_decay",
        "kl_factor",
        "norm_constraint",
        # Module templates
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
    ]
