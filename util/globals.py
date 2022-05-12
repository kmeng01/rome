import yaml
from pathlib import Path

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

RESULTS_DIR, DATA_DIR, STATS_DIR, COUNTERFACT_DIR, TFIDF_DIR, HPARAMS_DIR = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["COUNTERFACT_DIR"],
        data["TFIDF_DIR"],
        data["HPARAMS_DIR"],
    ]
)
