import argparse

from thcovmat import debug


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("what")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    if "thcovmat".startswith(args.what):
        debug.try_thcovmat()
    elif "prescriptions".startswith(args.what):
        debug.try_prescriptions()
