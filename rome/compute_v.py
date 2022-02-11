from cgitb import lookup
from typing import Dict, Tuple

import numpy as np
from rome import repr_tools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .rome_hparams import ROMEHyperParams

MAX_NORM = 100


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Handle inputs
    inp_tok = tok(
        [request["prompt"].format(request["subject"])], return_tensors="pt"
    ).to("cuda")
    prompt_template_kl = "{} is a"
    inp_kl_tok = tok(
        [prompt_template_kl.format(request["subject"])], return_tensors="pt"
    ).to("cuda")

    # Get vocab ID of new target token
    target_id = tok([request["target_new"]["str"]])["input_ids"][0][0]

    # Compute indices of the tokens where the fact is looked up
    lookup_idx = find_fact_lookup_idx(
        request["prompt"], request["subject"], tok, hparams.fact_token
    )
    lookup_idx_kl = find_fact_lookup_idx(
        prompt_template_kl, request["subject"], tok, hparams.fact_token
    )

    # Finalize rewrite and loss layers
    if layer == model.config.n_layer - 1:
        layer -= 1
        print(
            f"Note: reducing rewrite layer to {layer}. "
            f"Rewriting at layer {layer + 1} (the last layer) will have no effect."
        )
    print(f"Rewrite layer is {layer}")
    loss_layer = max(hparams.v_loss_layer, layer + 1)
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init = None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            if target_init is None:
                print("Recording initial target vector")
                # This vector should be consistent across the batch dimension
                target_init = cur_out[0, lookup_idx].detach()
            cur_out[:, lookup_idx] += delta[None, :]

        return cur_out

    # Keep track of original output distribution over "[subject] is a"
    # Helps avoid essence drift
    kl_distr_init = None

    # Similar delta logic, but for the KL constraint prompt
    def edit_output_fn_kl(cur_out, cur_layer):
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            cur_out[:, lookup_idx_kl] += delta[None, :]
        return cur_out

    # Optimizers
    opt = torch.optim.Adam([delta], lr=hparams.v_lr, weight_decay=hparams.v_weight_decay)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        trace_dict_args = dict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
        )

        # Forward pass of rewriting prompt
        with nethook.TraceDict(
            retain_output=True,
            edit_output=edit_output_fn,
            **trace_dict_args,
        ) as tr:
            model(**inp_tok)

        # Forward pass of distribution consistency prompt
        with nethook.TraceDict(
            retain_output=False,
            edit_output=edit_output_fn_kl,
            **trace_dict_args,
        ) as _:
            kl_logits = model(**inp_kl_tok).logits
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits[:, -1, :], dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().requires_grad_(False)

        # Compute probability distribution over tokens @ the loss layer
        cur_final = torch.log_softmax(
            (
                ln_f(
                    tr[hparams.layer_module_tmp.format(loss_layer)]
                    .output[0][0][-1]
                    .unsqueeze(0)
                )
                @ lm_w
                + lm_b
            ).squeeze(),
            dim=0,
        )

        # Compute value of objective function
        l1, l2 = -cur_final[target_id], hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True
        )
        loss = l1 + l2
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(l1.item(), 3)} + {np.round(l2.item(), 3)} "
            f"prob of [{tok.decode([target_id])}] "
            f"{torch.exp(cur_final[target_id]).item()}"
        )
        if loss < 5e-2:
            break

        # Backpropagate
        loss.backward()
        opt.step()

    target = target_init + delta

    # Retrieve `x`, the current input to the 2nd MLP layer, and
    # `current`, the original output of the 2nd MLP layer.
    x, current = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - current) / torch.dot(x, left_vector)
    print(f"Representation norm delta: {(target - current).norm().item()}")
    print(f"Division Factor: {torch.dot(x, left_vector).item()}")

    # Clamping hack to avoid catastrophe
    right_vector *= min(right_vector.norm().item(), MAX_NORM) / right_vector.norm()

    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves representations for a word at the input and output of a
    particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if fact_token_strategy == "subject_last":
        context_info = dict(
            context_template=context_template,
            word=word,
        )
        l_input = repr_tools.get_repr_at_word_last_token(
            track="in", **context_info, **word_repr_args
        )
        l_output = repr_tools.get_repr_at_word_last_token(
            track="out", **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        context_info = dict(
            context=context_template.format(word),
            idxs=[-1],
        )
        l_input = repr_tools.get_repr_at_idxs(
            track="in", **context_info, **word_repr_args
        )
        l_output = repr_tools.get_repr_at_idxs(
            track="out", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input, l_output


def find_fact_lookup_idx(
    prompt: str, subject: str, tok: AutoTokenizer, fact_token_strategy: str
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif fact_token_strategy == "subject_last":
        ret = repr_tools.get_last_word_idx_in_template(
            tok=tok,
            context_template=prompt,
            word=subject,
        )[0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    print(
        f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
        tok.decode(tok(sentence)["input_ids"][ret]),
    )

    return ret
