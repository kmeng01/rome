## Running the Full Evaluation Suite
An evaluation script is provided at [`experiments/evaluate.py`](experiments/evaluate.py). At a high level, it auto-loads required evaluation materials, iterates through each record in the dataset, and dumps results for each run in a `.json`. You can run `python3 -m experiments.evaluate -h` for details about command-line flags.

For example, to evaluate ROME on GPT-2 XL using [default parameters](hparams/ROME/gpt2-xl.json), run:
```bash
python3 experiments.evaluate --alg_name=ROME --model_name=gpt2-xl --hparams_fname=gpt2-xl.json
```

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):
```bash
python3 -m experiments.summarize --dir_name=ROME --runs=run_000
```

Again, you can run `python3 -m experiments.summarize -h` for information on the command-line flags.

_Note: evaluation is currently only supported for methods that edit autoregressive HuggingFace models using the PyTorch backend. We are working on a set of general-purpose methods (usable on e.g. TensorFlow and without HuggingFace) that will be released soon._

## Integrating New Editing Methods

<!-- Say you have a new method `X` and want to benchmark it on CounterFact. Here's a checklist for evaluating `X`:
- The public method that evaluates a model on each CounterFact record is [`compute_rewrite_quality`](experiments/py/eval_utils.py); see [the source code](experiments/py/eval_utils.py) for details.
- In your evaluation script, you should call `compute_rewrite_quality` once with an unedited model and once with a model that has been edited with `X`. Each time, the function returns a dictionary. -->

Say you have a new method `X` and want to benchmark it on CounterFact. To integrate `X` with our runner:
- Subclass [`HyperParams`](util/hparams.py) into `XHyperParams` and specify all hyperparameter fields. See [`ROMEHyperParameters`](rome/rome_hparams.py) for an example implementation.
- Create a hyperparameters file at `hparams/X/gpt2-xl.json` and specify some default values. See [`hparams/ROME/gpt2-xl.json`](hparams/ROME/gpt2-xl.json) for an example.
- Define a function `apply_X_to_model` which accepts several parameters and returns (i) the rewritten model and (ii) a dictionary of original weight values for ones that were edited (in the format `{weight_name: original_weight_values}`). See [`rome/rome_main.py`](rome/rome_main.py) for an example.
- Add `X` to `ALG_DICT` in [`experiments/evaluate.py`](experiments/evaluate.py) by adding the line `"X": (XHyperParams, apply_X_to_model)`.

Finally, run the main script:
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