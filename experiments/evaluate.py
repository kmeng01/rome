import json
import os
import shutil
import torch
from pathlib import Path
from time import time
from typing import Union, Tuple

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfact import AttributeSnippets, CounterFactDataset
from counterfact.tfidf.tfidf_stats import get_tfidf_vectorizer

from util import nethook

# All methods
from rome import ROMEHyperParams, apply_rome_to_model
from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor

# Evaluation tools
from experiments.py.eval_utils import compute_rewrite_quality

# Load directory configurations
load_dotenv()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))
COUNTERFACT_DIR = Path(os.getenv("COUNTERFACT_DIR"))
TFIDF_DIR = Path(os.getenv("TFIDF_DIR"))
HPARAMS_DIR = Path(os.getenv("HPARAMS_DIR"))
DATA_DIR = Path(os.getenv("DATA_DIR"))

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    conserve_memory: bool,
    dir_name: str,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [int(str(x).split("_")[-1]) for x in alg_dir.iterdir()]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    ds = CounterFactDataset(COUNTERFACT_DIR, size=dataset_size_limit)
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(TFIDF_DIR) if not skip_generation_tests else None

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # Iterate through dataset
    for record in ds:
        case_id = record["case_id"]
        case_result_path = run_dir / f"case_{case_id}.json"
        if not case_result_path.exists():
            # Compute weight changes + record weights that changed
            start = time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            edited_model, weights_copy = apply_algo(
                model,
                tok,
                [record["requested_rewrite"]],
                hparams,
                copy=False,
                return_orig_weights=True,
                **args_conserve_memory,
            )
            exec_time = time() - start
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": compute_rewrite_quality(edited_model, tok, record, snips, vec),
            }

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            metrics["pre"] = compute_rewrite_quality(model, tok, record, snips, vec)

            print("Evaluation took", time() - start)

            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        dir_name=args.alg_name,
    )
