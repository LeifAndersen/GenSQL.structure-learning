#!/usr/bin/env python

import argparse
import pandas as pd
import yaml
import sys


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w+"),
        default=sys.stdout,
        metavar="PATH",
    )
    parser.add_argument("--data", type=argparse.FileType("r"), help="Path to raw CSV.")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        metavar="PATH",
    )
    parser.add_argument(
        "--params",
        type=argparse.FileType("r"),
        help="Path to params.yaml",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    params = yaml.safe_load(args.params)
    subsampling_config = params["sub_sample"] or {}
    seed = params.get("seed", 42)

    held_out_combinations = subsampling_config.get("according_to_columns", {})
    if held_out_combinations:
        bool_index = pd.Series([False] * len(df))
        for column, values in held_out_combinations.items():
            bool_index = bool_index | (~df[column].isin(values))
    else:
        bool_index = pd.Series([True] * len(df))

    N = subsampling_config.get("N", None)
    if N is not None:
        assert N <= sum(
            bool_index
        ), "held-out configurations yield less data than specified with N"
        held_in = df[bool_index].sample(N, random_state=seed)
        held_out = df[bool_index].loc[
            [i for i in df[bool_index].index if i not in held_in.index]
        ]

        held_in.to_csv(args.output, index=False)
        held_out.to_csv(f"{args.test_data_dir}/test.csv", index=False)
    else:
        df[bool_index].to_csv(args.output)
        df[bool_index].to_csv(f"{args.test_data_dir}/test.csv", index=False)
    df[~bool_index].to_csv(f"{args.test_data_dir}/test-shifted.csv", index=False)


if __name__ == "__main__":
    main()
