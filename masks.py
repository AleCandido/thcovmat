# coding: utf-8
"""Generate masks and normalizations.

Masks are provided explicitly for the 3x3 prescriptions, the usual ones, but
most of them are generalizable to an higher number of points, (squared or even
not, according to the prescriptions).

The generalized prescriptions are provided as separate functions.

Since prescriptions are realized through masks, and normalization are dependent
on mass weights, even weights different from 0 or 1 could be used.

"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import rich
import seaborn as sns


@dataclass
class Prescription:
    mask: np.ndarray
    name: Optional[str] = None
    f0: Optional[int] = None
    r0: Optional[int] = None

    def __post_init__(self):
        for i, zero in enumerate(("f0", "r0")):
            if getattr(self, zero) is None:
                setattr(self, zero, self.mask.shape[i] // 2)

        self.nullify_central()

    def __repr__(self) -> str:
        return repr(self.mask)

    def nullify_central(self):
        # set to null
        self.mask[self.f0, self.r0] = 0

    @classmethod
    def ren(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 3 point ren"
        prescr = cls(np.zeros(shape), name="Renormalization only", f0=f0, r0=r0)
        prescr.mask[prescr.f0] = 1
        prescr.nullify_central()
        return prescr

    @classmethod
    def fact(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 3 point fact"
        prescr = cls(np.zeros(shape), name="Factorization only", f0=f0, r0=r0)
        prescr.mask[:, prescr.r0] = 1
        prescr.nullify_central()
        return prescr

    @classmethod
    def sum(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 3 point correlated"
        prescr = cls(np.zeros(shape), name="Fully correlated", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask, 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def antisum(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 3 point correlated"
        prescr = cls(np.zeros(shape), name="Fully anti-correlated", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask[::-1], 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def christ(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 5 point"
        el1 = cls.ren(shape, f0=f0, r0=r0)
        el2 = cls.fact(shape, f0=f0, r0=r0)
        return cls(np.logical_or(el1.mask, el2.mask) * 1.0, name="Christ", f0=f0, r0=r0)

    @classmethod
    def standrews(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 5 point correlated"
        el1 = cls.sum(shape, f0=f0, r0=r0)
        el2 = cls.antisum(shape, f0=f0, r0=r0)
        return cls(
            np.logical_or(el1.mask, el2.mask) * 1.0, name="St Andrews", f0=f0, r0=r0
        )

    @classmethod
    def tridiag(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 5 point correlated"
        prescr = cls(np.zeros(shape), name="Tridiagonal", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask, 1)
        np.fill_diagonal(prescr.mask[1:], 1)
        np.fill_diagonal(prescr.mask[:, 1:], 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def antitridiag(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 5 point correlated"
        prescr = cls(np.zeros(shape), name="Anti-tridiagonal", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask[::-1], 1)
        np.fill_diagonal(prescr.mask[::-1, 1:], 1)
        np.fill_diagonal(prescr.mask[-2::-1, :], 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def incoherent(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        "a.k.a. 9 point"
        prescr = cls(np.ones(shape), name="Fully incoherent", f0=f0, r0=r0)
        return prescr
    """Create integer masks' dictionary.

    Note
    ----
    The central scale is always enabled, but actually is never contributing:
    since the vector is a vector of shifts, the central one is always null, even
    without masking it.

    """
    prescriptions = ["3", "3b", "3c", "3cb", "5", "5b", "7", "7b", "9"]
    masks = {prescr: np.zeros((3, 3)) for prescr in prescriptions}

    masks["9"][:] = 1
    masks["3"][1] = 1
    masks["3b"][:, 1] = 1
    np.fill_diagonal(masks["3c"], 1)
    np.fill_diagonal(masks["3cb"][::-1], 1)
    masks["5b"] = np.logical_or(masks["3c"], masks["3cb"]) * 1
    masks["5"] = np.logical_or(masks["3"], masks["3b"]) * 1
    masks["7"] = np.logical_or(masks["5"], masks["3c"]) * 1
    masks["7b"] = np.logical_or(masks["5"], masks["3cb"]) * 1

    return masks


def s(mask: np.ndarray) -> int:
    s = 0

    if mask.sum() > 1:
        s += 1
    if any(
        any(mask.sum(axis) > 1) and all(mask.sum((axis + 1) % 2)) for axis in (0, 1)
    ):
        s += 1

    return s


def m(mask: np.ndarray) -> int:
    return np.sum(mask) - 1


if __name__ == "__main__":
    masks = prescriptions_masks()

    for prescr, mask in masks.items():
        rich.print(f"[green b]{prescr}[/], m: {m(mask)}, s: {s(mask)}")
        rich.print(*(f"    {line}" for line in str(mask).splitlines()), sep="\n")
