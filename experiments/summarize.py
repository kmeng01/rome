import collections
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

# Load directory configurations
load_dotenv()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR"))


def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (RESULTS_DIR / dir_name if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        for case_file in files:
            with open(case_file, "r") as f:
                data = json.load(f)

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            cur_sum["time"].append(data["time"])

            for prefix in ["pre", "post"]:
                # Probability metrics for which new should be lower (better) than true
                for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                    sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] > x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                                for x in data[prefix][key]
                            ]
                        )
                    )

                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                cur_sum[sum_key_discrete].append(
                    np.mean(
                        [x["target_true"] < x["target_new"] for x in data[prefix][key]]
                    )
                )
                cur_sum[sum_key_cont].append(
                    np.mean(
                        [
                            np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                            for x in data[prefix][key]
                        ]
                    )
                )

                # Numerical metrics that can be directly averaged
                for key in ["ngram_entropy", "reference_score", "essence_score"]:
                    if (
                        key in data[prefix] and data[prefix][key] is not None
                    ):  # Generation may have been skipped during eval
                        cur_sum[f"{prefix}_{key}"].append(data[prefix][key])

        if len(cur_sum) == 0:
            continue

        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)

        cur_sum.update(metadata)
        print(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
