from typing import TypeVar

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = TypeVar("T")


def to_device(t: T) -> T:
    return t.to_device(DEVICE)


class DotDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
