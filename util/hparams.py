import json


class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    KEYS = []

    def __init__(self, *args, **kwargs):
        for k in self.KEYS:
            self.__dict__[k] = None
        for k, v in kwargs.items():
            if k not in self.__dict__:
                raise KeyError(f"{k} not recognized")
            self.__dict__[k] = v

        assert all(
            v is not None for _, v in self.__dict__.items()
        ), "Not all arguments were initialized."

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)
