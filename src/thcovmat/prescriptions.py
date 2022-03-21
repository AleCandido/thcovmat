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
        """Remove central value, since it's a zero shift."""
        # set to null
        self.mask[self.f0, self.r0] = 0

    @classmethod
    def ren(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 3 point renormalization."""
        prescr = cls(np.zeros(shape), name="Renormalization only", f0=f0, r0=r0)
        prescr.mask[prescr.f0] = 1
        prescr.nullify_central()
        return prescr

    @classmethod
    def fact(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 3 point factorization."""
        prescr = cls(np.zeros(shape), name="Factorization only", f0=f0, r0=r0)
        prescr.mask[:, prescr.r0] = 1
        prescr.nullify_central()
        return prescr

    @classmethod
    def sum(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 3 point correlated."""
        prescr = cls(np.zeros(shape), name="Fully correlated", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask, 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def antisum(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 3 point anti-correlated."""
        prescr = cls(np.zeros(shape), name="Fully anti-correlated", f0=f0, r0=r0)
        np.fill_diagonal(prescr.mask[::-1], 1)
        prescr.nullify_central()
        return prescr

    @classmethod
    def christ(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 5 point."""
        el1 = cls.ren(shape, f0=f0, r0=r0)
        el2 = cls.fact(shape, f0=f0, r0=r0)
        return cls(np.logical_or(el1.mask, el2.mask) * 1.0, name="Christ", f0=f0, r0=r0)

    @classmethod
    def standrews(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 5 point correlated."""
        el1 = cls.sum(shape, f0=f0, r0=r0)
        el2 = cls.antisum(shape, f0=f0, r0=r0)
        return cls(
            np.logical_or(el1.mask, el2.mask) * 1.0, name="St Andrews", f0=f0, r0=r0
        )

    @classmethod
    def tridiag(
        cls, shape: Sequence[int], f0: Optional[int] = None, r0: Optional[int] = None
    ):
        """a.k.a. 7 point correlated."""
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
        """a.k.a. 7 point anti-correlated."""
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
        """a.k.a. 9 point."""
        prescr = cls(np.ones(shape), name="Fully incoherent", f0=f0, r0=r0)
        return prescr

    @property
    def s(self) -> int:
        """Number of independent scales."""
        m = self.mask.copy()

        m[self.f0, self.r0] = 1
        return np.log(np.sum(m**2)) / np.log(np.sum(m.shape) / 2)

    @property
    def m(self) -> int:
        """Number of prescription's points.

        Possibly weighted, for non-binary masks.

        """
        return np.sum(self.mask)

    @property
    def norm(self) -> float:
        """Normalization for given mask.

        Note
        ----
        Do **not** use directly this normalization for the theory covmariance
        matrix construction, since it ignores the presence of a further
        renormalization scale in the construction process.
        **Use instead** :meth:`N`.

        Returns
        -------
        float
            normalization for the given mask, i.e. for 1 factorization scale and
            1 renormalization scale

        """
        return self.s / self.m

    @property
    def N(self) -> float:
        """Theory covmat whole normalization.

        In the construction of the theory covmariance matrix, in order to make
        blocks uniform, 2 renormalization scales are always considered
        non-trivial (that is true for off-diagonal blocks, and enforced
        on-diagonal).
        In order to account for the 2 renormalization scales, the normalization
        for the mask only (i.e. :meth:`norm`) is not enough, but a division by
        the further independent renormalization scale is needed.

        Returns
        -------
        float
            the actual normalization to be used for the theory covmat

        """
        return self.norm / self.mask.shape[1]


def nbym(n: int = 3, m: Optional[int] = None) -> dict[str, Prescription]:
    """Create integer masks' dictionary.

    Note
    ----
    The central scale is always enabled, but actually is never contributing:
    since the vector is a vector of shifts, the central one is always null, even
    without masking it.

    Returns
    -------
    dict
      a dictionary with all the different prescriptions for the nxn scales
      (keys are specific for the 3x3 case)

    """
    if m is None:
        m = n

    prescriptions = {}

    prescriptions["3"] = Prescription.ren((n, m))
    prescriptions["3b"] = Prescription.fact((n, m))
    prescriptions["3c"] = Prescription.sum((n, m))
    prescriptions["3cb"] = Prescription.antisum((n, m))
    prescriptions["5"] = Prescription.christ((n, m))
    prescriptions["5b"] = Prescription.standrews((n, m))
    prescriptions["7"] = Prescription.tridiag((n, m))
    prescriptions["7b"] = Prescription.antitridiag((n, m))
    prescriptions["9"] = Prescription.incoherent((n, m))

    return prescriptions
