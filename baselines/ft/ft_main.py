import torch
from copy import deepcopy
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ft_hparams import FTHyperParams
from util import nethook


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) the weights that changed
    """

    deltas = execute_ft(model, tok, request, hparams)
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: FTHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing FT algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Define inputs
    inputs = tok([request["prompt"].format(request["subject"])], return_tensors="pt").to(
        "cuda"
    )
    target_id = tok([request["target_new"]["str"]])["input_ids"][0][0]

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()], lr=hparams.lr, weight_decay=hparams.weight_decay
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights.keys()

    # Update loop: intervene at layers simultaneously
    for it in range(hparams.num_steps):
        opt.zero_grad()
        probs = torch.nn.functional.log_softmax(model(**inputs).logits[0, -1, :], dim=0)
        loss = -probs[target_id]
        print(
            f"loss {loss.item()} prob of [{tok.decode([target_id])}] "
            f"{torch.exp(probs[target_id]).item()}"
        )

        if not loss.item() < 1e-2:
            loss.backward()
            opt.step()

        if type(hparams.norm_constraint) is float:
            eps = hparams.norm_constraint
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = torch.clamp(
                        v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                    )

        if loss.item() < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights.keys()}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas
