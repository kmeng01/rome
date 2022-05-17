import typing
from torch.utils.data import Dataset
from pathlib import Path
import json


class CounterFactDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = 10000,
    ):
        with open(Path(data_dir) / "counterfact.json", "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
