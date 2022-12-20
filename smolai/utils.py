from typing import TypeVar

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = TypeVar("T")


def to_device(t: T) -> T:
    return t.to(DEVICE)


def advance(cb_gen, requires_grad):
    try:
        if requires_grad:
            return next(cb_gen)
        else:
            with torch.no_grad():
                return next(cb_gen)
    except StopIteration:
        pass
