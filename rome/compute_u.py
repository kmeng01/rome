import os
from pathlib import Path
from typing import Dict

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from rome import repr_tools
from util.generate import generate_fast

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}
context_templates_cache = None

# Load directory configurations
load_dotenv()
STATS_DIR = Path(os.getenv("STATS_DIR"))


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """
    global context_templates_cache

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if hparams.fact_token == "subject_last":
        # Sample some prefixes to get the contextual embedding of subject
        word = request["subject"]
        # context_templates = ["{}"]
        if context_templates_cache is None:
            context_templates_cache = [
                x + " {}"
                for x in sum(
                    (
                        generate_fast(
                            model,
                            tok,
                            ["The"],
                            n_gen_per_prompt=n_gen,
                            max_out_len=length,
                        )
                        for length, n_gen in [(2, 20), (5, 20), (10, 10)]
                    ),
                    [],
                )
            ] + ["{}"]

            print(f"Cached context templates {context_templates_cache}")

        print(f"Selected u projection token {word}")
        cur_repr = torch.stack(
            [  # TODO batch this to drastically improve performance
                repr_tools.get_repr_at_word_last_token(
                    context_template=templ, word=word, **word_repr_args
                )
                for templ in context_templates_cache
            ],
            dim=0,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_repr_at_idxs(
            context=request["prompt"].format(request["subject"]),
            idxs=[-1],
            **word_repr_args,
        )
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply covariance estimate
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
