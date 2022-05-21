import yaml
from pathlib import Path

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

RESULTS_DIR, DATA_DIR, STATS_DIR, COMPILED_DATASETS_DIR, TFIDF_DIR, HPARAMS_DIR, REMOTE_ROOT_URL = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["COMPILED_DATASETS_DIR"],
        data["TFIDF_DIR"],
        data["HPARAMS_DIR"],
        data["REMOTE_ROOT_URL"]
    ]
)
