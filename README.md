# Rank-One Model Editing (ROME)

This repository provides an implementation of Rank-One Model Editing (ROME) on auto-regressive transformers (GPU-only).
We currently support OpenAI's GPT-2 S/M/L/XL (up to 1.5B) and EleutherAI's GPT-J (6B). The release of a 20B GPT-like model from EleutherAI is expected soon; we hope to support it ASAP.

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

We compare ROME against several state-of-the-art model editors. All are implemented in [baselines/](baselines) in their respective folders. Implementations are not our own; they are adapted slightly to plug into our evaluation system.
- Knowledge Neurons (KN): Dai et al. [[Code]](https://github.com/EleutherAI/knowledge-neurons) [[Paper]](https://arxiv.org/abs/2104.08696)
- Knowledge Editor (KE): De Cao et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2104.08164)
- Model Editor Networks with Gradient Decomposition (MEND): Mitchell et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2110.11309)

### Running the Full Evaluation Suite
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

Description coming soon!

## How to Cite

```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Knowledge in GPT},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={arXiv preprint arXiv:2202.05262},
  year={2022}
}
```