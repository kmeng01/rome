# Rank-One Model Editing (ROME)

This repository provides an implementation of Rank-One Model Editing (ROME) on auto-regressive transformers (GPU-only).
We currently support OpenAI's GPT-2 XL (1.5B) and EleutherAI's GPT-J (6B). The release of a 20B GPT-like model from EleutherAI is expected soon; we hope to support it ASAP.

Feel free to open an issue if you find any problems; we are actively developing this repository and will monitor tickets closely.

[![Colab ROME Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb)

<p align="center">
    <img src="https://rome.baulab.info/images/eiftower-crop.svg" alt="causal tracing GIF" width="425px" />
</p>

## Table of Contents
1. [Installation](#installation)
2. [Causal Tracing](#causal-tracing)
3. [Rank-One Model Editing (ROME)](#rank-one-model-editing-rome-1)
4. [CounterFact Dataset](#counterfact)
5. [Evaluation](#evaluation)
6. [How to Cite](#how-to-cite)

## Installation

We recommend `conda` for managing Python, CUDA, and PyTorch-related dependencies, and `pip` for everything else. To get started, simply install `conda` and run:
```bash
./scripts/setup_conda.sh
```

## Causal Tracing

[`notebooks/causal_trace.ipynb`](notebooks/causal_trace.ipynb) demonstrates Causal Tracing, which can be modified to apply tracing to the processing of any statement.

<p align="center">
    <img src="https://thevisible.net/u/davidbau/romeweb/small-fast-ct-animation.gif" alt="causal tracing GIF" width="550px" />
</p>

## Rank-One Model Editing (ROME)

<!-- We provide a simple interactive notebook demonstrating ROME. -->

<!-- ### Second-Moment Key Statistics

**warning this is probably wrong; fixing later.**

First, key statistics must be collected. The `rome` package contains a `layer_stats` module for computing and caching key statistics. See [rome/layer_stats.py](rome/layer_stats.py) for additional flags, but the basic logic can be executed with the following commands:

GPT-2 XL:
```bash
python -m rome.layer_stats --layer_num=17 --model_name=gpt2-xl
```

GPT-J:
```bash
python -m rome.layer_stats --layer_num=10 --model_name=EleutherAI/gpt-j-6B
```

### ROME Model Rewriting -->

[`notebooks/rome.ipynb`](notebooks/rome.ipynb) demonstrates ROME. The API is simple; one simply has to specify a *requested rewrite* of the following form:

```python
request = {
    "prompt": "{} plays the sport of",
    "subject": "LeBron James",
    "target_new": {
        "str": "football"
    }
}
```

Several similar examples are included in the notebook.

## CounterFact

Description coming soon!

## Evaluation

### Paper Baselines

We compare ROME against several open sourced state-of-the-art model editors. All are implemented in [`baselines/`](baselines) in their respective folders. Implementations are not our own; they are adapted slightly to fit our evaluation system.
- Knowledge Neurons (KN): Dai et al. [[Code]](https://github.com/EleutherAI/knowledge-neurons) [[Paper]](https://arxiv.org/abs/2104.08696)
- Knowledge Editor (KE): De Cao et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2104.08164)
- Model Editor Networks with Gradient Decomposition (MEND): Mitchell et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2110.11309)

### Running the Full Evaluation Suite
[`experiments/evaluate.py`](experiments/evaluate.py) contains evaluation code for all methods presented in the paper. At a high level, it auto-loads required evaluation materials, iterates through each record in the dataset, and dumps results for each run in a `.json`. Run `python3 -m experiments.evaluate -h` for details on command-line flags.

For example, if you'd like to fully evaluate ROME on GPT-2 XL using [default parameters](hparams/ROME/gpt2-xl.json), you can run:
```bash
python3 experiments.evaluate --alg_name=ROME --model_name=gpt2-xl --hparams_fname=gpt2-xl.json
```

Evaluation is currently only supported for PyTorch-based methods that edit HuggingFace `AutoModelForCausalLM` models. We are working on a set of general-purpose methods (useable on e.g. TensorFlow non-HuggingFace) that will be released soon.

## Integrating and Evaluating New Editing Methods

<!-- Say you have a new method `X` and want to benchmark it on CounterFact. Here's a checklist for evaluating `X`:
- The public method that evaluates a model on each CounterFact record is [`compute_rewrite_quality`](experiments/py/eval_utils.py); see [the source code](experiments/py/eval_utils.py) for details.
- In your evaluation script, you should call `compute_rewrite_quality` once with an unedited model and once with a model that has been edited with `X`. Each time, the function returns a dictionary. -->

Say you have a new method `X` and want to benchmark it on CounterFact. We already have a runner that instruments the evaluation loop at [`experiments/evaluate.py`](experiments/evaluate.py). To use it:
- Subclass [`HyperParams`](util/hparams.py) into `XHyperParams` and specify all hyperparameter fields. See [`ROMEHyperParameters`](rome/rome_hparams.py) for an example implementation.
- Create a hyperparameters file at `hparams/X/gpt2-xl.json` and specify some default values. See [`hparams/ROME/gpt2-xl.json`](hparams/ROME/gpt2-xl.json) for an example.
- Define a function `apply_X_to_model` which accepts several parameters and returns (i) the rewritten model and (ii) a dictionary of original weight values for ones that were edited (in the format `{weight_name: original_weight_values}`). See [`rome/rome_main.py`](rome/rome_main.py) for an example.
- Add `X` to `ALG_DICT` in [`experiments/evaluate.py`](experiments/evaluate.py) by adding the line `"X": (XHyperParams, apply_X_to_model)`.

Finally, run the main script!
```bash
python3 experiments.evaluate --alg_name=X --model_name=gpt2-xl --hparams_fname=gpt2-xl.json
```

<!-- 
Each method is customizable through a set of hyperparameters. For ROME, they are defined in `rome/hparams.py`. At runtime, you must specify a configuration of hyperparams through a `.json` file located in `hparams/<method_name>`. Check out [`hparams/ROME/default.json`](hparams/ROME/default.json) for an example.

At runtime, you must specify two command-line arguments: the method name, and the filename of the hyperparameters `.json` file.
```bash
python3 -m experiments.evaluate --alg_name=ROME --hparams_fname=default.json
```

Results from each run are stored in a directory of the form `results/<method_name>/run_<run_id>`.

Running the following command will yield `dict` run summaries:
```bash
python3 -m experiments/summarize --alg_name=ROME --run_name=run_001
``` -->

## How to Cite

```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Knowledge in GPT},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={arXiv preprint arXiv:2202.05262},
  year={2022}
}
```
