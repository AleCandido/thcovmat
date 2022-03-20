import argparse

import rich

from .. import prescriptions
from .. import thcovmat
from . import out


def try_prescriptions():
    prescrs = prescriptions.nbyn(11)

    for name, prescr in prescrs.items():
        if name == "7b":
            rich.print(f"[b white] {name}")
            out.plot_prescription(prescr)


def try_thcovmat():
    raw = thcovmat.raw_shifts((1000, 50, 400, 2000, 160, 720, 1000))
    shifts = thcovmat.shifts_vec(raw)
    mat = thcovmat.thcovmat(shifts)

    print(mat.shape)
    #  out.block_plot(mat, 30, "thcovmat.png")
    out.block_plot(mat, 30)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("what")

    return parser.parse_args()


def cli():
    args = parse()

    if "thcovmat".startswith(args.what):
        try_thcovmat()
    elif "prescriptions".startswith(args.what):
        try_prescriptions()
