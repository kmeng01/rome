from typing import Dict, Tuple, List
from matplotlib.style import context
import numpy as np
from rome import repr_tools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .rome_hparams import ROMEHyperParams


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
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

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"])["input_ids"]

    # Compile list of rewriting x/y pairs
    rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:i])
        for context in context_templates
        for i in range(len(target_ids))
    ]
    rewriting_targets = [
        target_ids[i]
        for _ in range(len(context_templates))
        for i in range(len(target_ids))
    ]
    assert len(rewriting_prompts) == len(rewriting_targets)

    # Compile KL divergence prompts
    kl_prompts = ["{} is a"]

    # Compute rewriting inputs and outputs
    # Note that these have different lengths; KL prompts have no targets
    all_prompts = rewriting_prompts + kl_prompts
    opt_inputs = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    opt_targets = torch.tensor(rewriting_targets).to("cuda")

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**opt_inputs).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Gather output representation at last non-masked token
        # Careful to slice out the KL divergence prompts
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][:len(rewriting_prompts)]
        indices_to_gather = (
            (opt_inputs["attention_mask"][:len(rewriting_prompts)].sum(1) - 1)
            .unsqueeze(1)
            .repeat(1, full_repr.size(-1))
            .unsqueeze(1)
        )
        gathered_reprs = torch.gather(full_repr, 1, indices_to_gather).squeeze(1)

        # Compute probability distribution over tokens @ the loss layer
        log_dist = torch.log_softmax(ln_f(gathered_reprs) @ lm_w + lm_b, dim=1)

        # Compute value of objective functions
        nll_loss_each = -torch.gather(
            log_dist, 1, opt_targets.unsqueeze(0)
        )
        # print(torch.exp(-nll_loss_each))
        nll_loss = nll_loss_each.sum()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    # TODO cur_output is redundant with target_init
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
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
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
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

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
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
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
