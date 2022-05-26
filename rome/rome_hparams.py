from util.hparams import HyperParams


class ROMEHyperParams(HyperParams):
    KEYS = [
        # Method
        "layers",
        "fact_token",  # [last, subject_first, subject_last, subject_first_after_last]
        "v_num_grad_steps",
        "v_lr",
        "v_loss_layer",
        "v_weight_decay",
        "clamp_norm_factor",
        "kl_factor",
        "mom2_adjustment",
        # Module templates
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
        # Statistics
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
    ]
