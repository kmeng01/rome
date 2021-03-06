import os
from copy import deepcopy
from typing import Dict

import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
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
        request: Dict,
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

        request_rewrite = deepcopy(request)

        target = " " + request_rewrite["target_new"]["str"]
        sentence = request_rewrite["prompt"].format(request_rewrite["subject"]) + target
        target_tokens = self.tokenizer(target)["input_ids"]
        tokens = torch.tensor(self.tokenizer(sentence)["input_ids"])[None]
        label_tokens = tokens.clone()
        label_tokens[0][: -len(target_tokens)] = -100

        edit_inner = dict(
            input_ids=tokens.clone().cuda(),
            attention_mask=torch.ones_like(tokens).cuda(),
            labels=label_tokens.clone().cuda(),
        )
        cond = dict(
            input_ids=tokens.clone().cuda(),
            attention_mask=torch.ones_like(tokens).cuda(),
        )
        edited_model, model_info = self.alg.edit(edit_inner, cond, return_factors=True)

        mean_grads = {
            n: torch.einsum(f"bi,bj->ij", x, delta)
            for n, (x, delta) in model_info["factors"].items()
        }
        edit_lrs = self.alg.edit_lrs.detach().clone()

        model = deepcopy(self.model) if copy else self.model
        w_backups = {}

        with torch.no_grad():
            for lr, (n, g) in zip(edit_lrs, mean_grads.items()):
                cur_weight = nethook.get_parameter(model, n)
                if return_orig_weights and n not in w_backups:
                    w_backups[n] = cur_weight.detach().clone()

                upd_matrix = lr * g * hparams.lr_scale
                if upd_matrix.shape != cur_weight.shape:
                    upd_matrix = upd_matrix.T

                cur_weight[...] += upd_matrix

        return model, w_backups
