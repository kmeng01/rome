from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .kn_hparams import KNHyperParams
from .knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type


def apply_kn_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KNHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_name),
        device="cuda",
    )
    request_rewrite = deepcopy(request)
    text = [request_rewrite["prompt"].format(request_rewrite["subject"])]
    ground_truth = request_rewrite["target_true"]["str"]
    target = request_rewrite["target_new"]["str"]

    kn.model = kn.model.to(kn.device)
    refined_neurons = kn.get_refined_neurons(
        text,
        ground_truth,
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )

    results_dict, unpatch_fn = kn.edit_knowledge(
        text[0],
        target=target,
        neurons=refined_neurons,
        undo_modification=False,
    )
    updated_model = deepcopy(kn.model)
    with torch.no_grad():
        unpatch_fn()
    return updated_model, {}
