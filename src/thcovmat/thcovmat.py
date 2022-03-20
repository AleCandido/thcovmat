"""`numpy` implementation of theory covmat generation.

Comparison with the old method:

- performance: <5s against ~1h
- <30 sloc, unique for all prescriptions (same amount was needed to prepare
  before applying prescription specific code)
- prescriptions: implemented only as mask on the raw shifts (to mask the matrix
  second and third defined by the 2nd and 3rd dimensions), and dividing by the
  suitable normalization
- the mask is the exact same of the one in the picture, no more
- it scales for a different number of scale variations, 3 can be replaced by
  whatever, independently for fact scale and ren scale
- the output is immediately a `np.ndarray`, with a reasonable dimension in
  memory even for ~5k points (and in any case it's easy to produce eachy block
  separately, it sufficiently to yield one by one the elements produced by
  :func:`thcovmat`)

"""
from typing import Iterable

import numpy as np


def raw_shifts(sizes: Iterable[int]) -> list[np.ndarray]:
    """Generate a sample of theory shifts.

    I sample from a gaussian, with:

    - mean equal to 10 times the index of the dataset
    - variance equal to the index of the dataset

    In this way each batch will its own unique signature.

    Note
    ----
    Indices are zero-based.

    Parameters
    ----------
    sizes : iterable of int
      the sizes of the individual datasets to generate

    Returns
    -------
    list[np.ndarray]
        a list of batches, each one corresponding to pseudo-data for a
        pseudo-experimental set; the dimension of each batch is ``(n, 3, 3)``,
        where:

        - ``n`` is the size of each batch (a.k.a. dataset)
        - the second dimension is factorization scale
        - the third dimension is the renormalization scale specific for the
          process

    """
    return [
        np.random.normal(loc=10 * i, scale=i, size=(batch, 3, 3))
        for i, batch in enumerate(sizes)
    ]


def shifts_vec(raw: list[np.ndarray]) -> list[np.ndarray]:
    """Pump more dimensions into shifts.

    Since renormalization scale for each process has a different meaning, the
    most intrinsic and intuitive way to represent this difference is to put them
    on different dimensions:

    - a renormalization dimension is non-trivial only for the process that scale
      is relative to
    - a trivial dimension has the meaning of not affecting that datum
    - trivial dimension can be exploited by `numpy` with the broadcasting
      semantic

    Parameters
    ----------
    raw : list[np.ndarray]
        sequence of raw shifts, of dimension ``(n,3,3)``, like those generated
        by :func:`raw_shifts`

    Returns
    -------
    list[np.ndarray]
        a list of batches, blown up with new dimensions to separate on different
        dimensions the renormalization scales related to different processes

    """
    upgraded = []
    n = len(raw)
    dims = list(np.arange(n) + 2)
    for i, batch in enumerate(raw):
        newdims = dims.copy()
        newdims.remove(i + 2)
        upgraded.append(np.expand_dims(batch, newdims))

    return upgraded


def thcovmat(shifts: list[np.ndarray]) -> np.ndarray:
    """Generate theory covariance matrix from upgraded vector of shifts.

    Exploit `numpy` broadcasting to apply Eq.(4.2) of arXiv:1906.10698
    literally.

    The prescription used is always the 9-point one, so the other prescriptions
    have to be implemented on the input and the output of this funcions, i.e.:

    - the input has to be masked, to replace elements that should be missing
      with 0s (it is trivial that the effect is the same)
    - an overall normalization

    Note
    ----
    In order to apply exactly Eq.(4.2), all the renormalization scale dimensions
    should be non-trivial, and the broadcasting semantics should be implemented
    for all dimensions and on the vector itself, such that a single rectangular
    `np.ndarray` is obtained.

    This of course is very inefficient for memory and operations, since
    :math:`n-2` renormalization scales are always not involved while generating
    a block for 2 given processes.

    So for the :math:`n-2` scales it would correspond to an overall degeneracy
    factor that would simplify with normalization, so we can simply skip (in
    `numpy` this is done by the contraction on a dimension of size 1 for both
    arrays, resulting in the removal of that dimension).

    The remaining scales are at most 2, but not always 2, changing from
    on-diagonal or off-diagonal blocks. The difference has to be compensated
    with as a further degeneracy factor for the on-diagonal blocsk.

    A minimal example is the case of only to process: there would be two shift
    arrays, one each, of shapes ``[n1, nf, nr, 1]`` and ``[n2, nf, 1, nr]``. The
    internal dimensions on data is irrelevant, so we can drop, as well as the
    factorization scale one (that is always the same).
    For definiteness let's consider ``nr = 3``, so we'll have two shifts arrays
    of shapes ``[3,1]`` and ``[1, 3]``.

    - off-diagonal: the contraction is done with ``[3,1]`` and ``[1,3]``, and
      both the 1 dimensions are broadcasted to 3 before contraction, leading to
      a 9 elements contraction
    - on-diagonal: a contraction is ``[3, 1]`` with ``[3, 1]``, so the second
      dimension is not broadcasted, leading to only 3 elements, but it should be
      according to the meaning of Eq.(4.2); for this reason, a degeneracy
      factor of 3 is multiplied, in order to make it homogeneous with the
      off-diagonal blocks


    Parameters
    ----------
    raw : list[np.ndarray]
        sequence of upgraded shifts, like those generated by :func:`shifts_vec`

    Returns
    -------
    np.ndarray
        matrix of "covariances" generated out of shifts, whose shape is ``(N,
        N)``, where ``N`` is the number of all data points (i.e. ``N =
        sum(len(s) for s in shifts)``)

    Raises
    ------
    ValueError
      if not all the processes have the same number of points for their own
      renormalization scales

    """
    blockmat = []
    sumdims = list(np.arange(len(shifts) + 1) + 2)

    # for each process there is only one non-trivial renormalization scale, so
    # the other dimensions have to be 1
    murs = np.array(shifts[0].shape[2:]).prod()
    if not all(np.array(proc_shift.shape[2:]).prod() == murs for proc_shift in shifts):
        raise ValueError(
            "All the different renormalization scales should have the"
            " same number of points"
        )

    for bi in shifts:
        blockrow = []
        for bj in shifts:
            degeneracy = murs if bi is bj else 1.0
            blockrow.append(
                np.einsum(bi, [0, *sumdims], bj, [1, *sumdims], [0, 1]) * degeneracy
            )
        blockmat.append(blockrow)

    return np.block(blockmat)
