"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_repr_at_word_token(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_template: str,
    word: str,
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_word_idx_in_template(tok, context_template, word, subtoken)
    return get_repr_at_idxs(
        model,
        tok,
        context_template.format(word),
        idxs,
        layer,
        module_template,
        track,
    )


def get_repr_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context: str,
    idxs: List[int],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    assert track in {"in", "out"}
    tin, tout = (
        (track == "in"),
        (track == "out"),
    )
    module_name = module_template.format(layer)
    context_tok = tok([context], return_tensors="pt").to(
        next(model.parameters()).device
    )

    with torch.no_grad():
        with nethook.Trace(
            model,
            module_name,
            retain_input=tin,
            retain_output=tout,
        ) as tr:
            model(**context_tok)

    # cur_repr is already detached due to torch.no_grad()
    cur_repr = tr.input if track == "in" else tr.output
    cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr

    return torch.stack([cur_repr[0, i, :] for i in idxs], dim=1).mean(1)


def get_word_idx_in_template(
    tok: AutoTokenizer, context_template: str, word: str, subtoken: str
) -> int:
    """
    Given a template string `context_template` with *one* format specifier
    (e.g. "{} plays basketball") and a word `word` to be substituted into the
    template, computes the post-tokenization index of `word`'s last token in
    `context_template`.
    """

    assert (
        context_template.count("{}") == 1
    ), "We currently do not support multiple fill-ins for context"

    fill_idx = context_template.index("{}")
    prefix, suffix = context_template[:fill_idx], context_template[fill_idx + 2 :]
    if len(prefix) > 0:
        assert prefix[-1] == " "
        word = f" {word.strip()}"
        prefix = prefix[:-1]

    prefix_tok, word_tok, suffix_tok = tok([prefix, word, suffix])["input_ids"]

    if subtoken == "last" or subtoken == "first_after_last":
        return [
            len(prefix_tok)
            + len(word_tok)
            - (1 if subtoken == "last" or len(suffix_tok) == 0 else 0)
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
        ]
    elif subtoken == "first":
        return [len(prefix_tok)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")
