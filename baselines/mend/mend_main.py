import os
from copy import deepcopy
from typing import Dict, List

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *

from .algs.mend import MEND
from .mend_hparams import MENDHyperParams


class MendRewriteExecutor:
    method_name = "MEND"

    def __init__(self):
        self.is_init = False

    def init_model(self, model, tok, params):
        train_ds = (
            "counterfact-" if params.counterfact else ("zsre-" if params.zsre else "")
        )
        mini_string = "mini-" if params.mini else ""

        model_name = "gpt2-xl" if params.model_name == "gpt2-xl" else "gpt-j-6b"
        modelcode = "gpt2xl" if params.model_name == "gpt2-xl" else "gptj"
        model_filename = (
            f"mend-{mini_string}{params.n_toks}tok-{train_ds}{model_name}.pt"
        )
        model_dir = "baselines/mend/weights"

        os.makedirs(model_dir, exist_ok=True)
        if not os.path.isfile(f"{model_dir}/{model_filename}"):
            remote_url = f"{REMOTE_ROOT_URL}/data/weights/{model_filename}"
            print(f"Attemping to download from {remote_url}")
            torch.hub.download_url_to_file(remote_url, f"{model_dir}/{model_filename}")
        with hydra.initialize(config_path="config", job_name="run"):
            config = hydra.compose(
                config_name="config",
                overrides=[
                    "+alg=mend",
                    "+experiment=gen",
                    f"+model={modelcode}",
                    f"data.path=data/{params.n_toks}token/data/self_sample/",
                ],
            )

        def add_padding(tokenizer, model):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight.data[
                -1
            ] = model.transformer.wte.weight.data.mean(0)

        # Customize the gpt2xl and tokenizer
        self.model = model
        self.tokenizer = tok
        add_padding(self.tokenizer, self.model)

        # Load the trained MEND model
        self.alg = MEND(self.model, config, lambda: deepcopy(self.model))
        d = torch.load(f"{model_dir}/{model_filename}")
        self.alg.load_state_dict(
            {k.replace("gtn.", "mend."): v for k, v in d["model"].items()}
        )
        self.alg.cuda()

        # Disable unneeded gradients
        for n, p in self.model.named_parameters():
            if n not in config.model.inner_params:
                p.requires_grad = False
        self.is_init = True

    def reset_model(self):
        self.is_init = False
        del self.model, self.tokenizer, self.alg

    def apply_to_model(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MENDHyperParams,
        copy=False,
        return_orig_weights=False,
    ):
        """
        Given a request, for example
        {'prompt': '{} has the position of',
         'subject': 'Charles Herman Helmsing',
         'relation_id': 'P39',
         'target_new': {'str': 'President', 'id': 'Q11696'},
         'target_true': {'str': 'bishop', 'id': 'Q29182'}}
        Returns a dictionary of numpy arrays that specifies
        how mend will change the weights of the model.
        """

        if not self.is_init:
            self.init_model(model, tok, hparams)

        weights_copy = {}
        model = deepcopy(self.model) if copy else self.model

        # Define i/o
        targets = [
            (" " if request["target_new"]["str"][0] != " " else "")
            + request["target_new"]["str"]
            for request in requests
        ]
        sentences = [
            request["prompt"].format(request["subject"]) + targets[i]
            for i, request in enumerate(requests)
        ]

        # Tokenize
        sent_tok = self.tokenizer(sentences, padding=True, return_tensors="pt").to(
            "cuda"
        )
        target_tok = self.tokenizer(targets, padding=True, return_tensors="pt").to(
            "cuda"
        )

        # Define labels
        label_tok = deepcopy(sent_tok["input_ids"])
        for i in range(label_tok.size(0)):
            target_len = target_tok["attention_mask"][i].sum()
            padding_len = (
                sent_tok["input_ids"].size(1) - sent_tok["attention_mask"][i].sum()
            )
            label_tok[i][: -target_len - padding_len] = -100
            label_tok[i][label_tok[i] == self.tokenizer.pad_token_id] = -100

        # Run MEND
        edit_inner = dict(
            input_ids=sent_tok["input_ids"],
            attention_mask=sent_tok["attention_mask"],
            labels=label_tok,
        )
        cond = {k: sent_tok[k] for k in ["input_ids", "attention_mask"]}
        _, model_info = self.alg.edit(edit_inner, cond, return_factors=True)
        factors = {
            k + "." + n: v.detach().cpu().numpy()
            for k, pair in model_info["factors"].items()
            for n, v in zip("uv", pair)
        }
        # Also keep these learned LRs.
        factors["edit_lrs"] = self.alg.edit_lrs.detach().cpu().numpy()

        # Edit!
        d = factors
        torch_factors = {k: torch.tensor(v) for k, v in d.items()}
        eli = 0
        edit_lrs = torch_factors["edit_lrs"]

        with torch.no_grad():
            for n, p in model.named_parameters():
                uname, vname = f"{n}.u", f"{n}.v"
                if uname in torch_factors:
                    if return_orig_weights and n not in weights_copy:
                        weights_copy[n] = p.detach().clone()

                    if "gpt2" in hparams.model_name:
                        delta = torch_factors[uname].t() @ torch_factors[vname]
                    elif "gpt-j-6B" in hparams.model_name:
                        delta = torch_factors[vname].t() @ torch_factors[uname]
                    else:
                        raise ValueError("Unknown model")
                    p.add_((delta * edit_lrs[eli] * hparams.lr_scale).to(p.device))
                    eli += 1

        return model, weights_copy
