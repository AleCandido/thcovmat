from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import rich
import seaborn as sns

from ..prescriptions import Prescription


def block_plot(covmat: np.ndarray, block_size: int, filename: Optional[str] = None):
    """Plot blocked heatmap of the given squared matrix.

    If no value is passed for the file name, an interactive plot is created and
    shown.

    Parameters
    ----------
    covmat : np.ndarray
      the squared matrix to plot
    block_size : int
      the size of the square blocks in which elements of `covmat` are summed
    filename : str or None
      the name of the file to save

    Raises
    ------
    AssertionError
      if the matrix is not squared

    """

    # the matrix has to be squared
    assert covmat.shape[0] == covmat.shape[1]
    dim = covmat.shape[0]

    # coarse grain it in nxn blocks
    n = block_size
    indices = np.arange(dim - 1)[::n]
    blocked = np.add.reduceat(np.add.reduceat(covmat, indices), indices, axis=1)
    # drop leftover
    last = indices[-1]
    if last != dim - n:
        blocked = blocked[:-1, :-1]

    sns.heatmap(blocked)
    plt.xticks()
    plt.yticks()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_prescription(prescr: Prescription):
    plt.title(prescr.name)
    sns.heatmap(prescr.mask)
    plt.show()


def pprint_prescription(prescr: Prescription):
    rich.print(f"[green b]{prescr.name}[/], m: {prescr.m}, s: {prescr.s}")
    rich.print(*(f"    {line}" for line in str(prescr).splitlines()), sep="\n")
