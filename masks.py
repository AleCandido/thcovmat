# coding: utf-8
import numpy as np
import rich


def prescriptions_masks() -> dict[str, np.ndarray]:
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
