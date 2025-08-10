#!/usr/bin/env python

import argparse
import edn_format
import itertools
import json
import pandas
import sys
import numpy as np
import scipy.stats as stats
from collections import defaultdict

# Monkey patching this to work around https://github.com/scipy/scipy/pull/7838

if not hasattr(stats, "frechet_r"):
    stats.frechet_r = stats.weibull_max

if not hasattr(stats, "frechet_l"):
    stats.frechet_l = stats.weibull_min


def lin_reg(xs, ys):
    """Compute linear regression if we have data on both xs and ys"""
    if len(xs) > 0:  # Check whether nan mask return data
        r = stats.linregress(xs, ys)
        # Check if stats.linregress ran into numerical problems.
        if not (np.isnan(r.slope) or np.isnan(r.intercept)):
            return {
                "slope": r.slope,
                "intercept": r.intercept,
                "r-value": r.rvalue,
                "p-value": r.pvalue,
            }
    # Else: return a placeholder that works with subsequent dvc stages.
    return {
        "slope": 0.0,
        "intercept": 0.0,
        "r-value": 0.0,
        "p-value": 1.0,
    }


def chi_squared(xs, ys):
    """Compute chi-square statistic"""
    contingency = pandas.crosstab(xs, ys)
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
    except ValueError: ### XXX This is probably wrong!
        chi2, p, dof, expected = 0, 0, 0, 0
    return {"chi2": chi2, "p-value": p, "dof": dof}


def not_null(v):
    if isinstance(v, str):
        return True
    elif v is None:
        return False
    elif np.isnan(v):
        return False
    return True


def anova(df, c1, c2):
    """Compute One-way ANOVA"""
    samples = []
    for categorical_value in df[c1].unique():
        # Check whether this is actually a string category
        if not_null(categorical_value):
            samples.append(
                df[(df[c1] == categorical_value) & (~df[c2].isnull())][c2].values
            )
    F, p = stats.f_oneway(*samples) if len(samples) > 1 else np.nan, np.nan
    return {
        "F": F,
        "p-value": p,
    }


def main():
    description = "Outputs linear correlation info for all pairs of numerical columns."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--data",
        type=argparse.FileType("r"),
        help="Path to CSV used to generate correlations.",
    )
    parser.add_argument(
        "--schema", type=argparse.FileType("r"), help="Path to base schema."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w+"),
        help="Path to which correlation information will be written as a csv.",
        default=sys.stdout,
    )

    args = parser.parse_args()
    df = pandas.read_csv(args.data)

    schema = edn_format.loads(args.schema.read(), write_ply_tables=False)

    pairs = itertools.combinations(df.columns, 2)

    results = defaultdict(dict)

    for c1, c2 in pairs:
        stattypes = {schema[c1].name, schema[c2].name}
        c1_vals = df[c1].values
        c2_vals = df[c2].values
        if "ignore" in stattypes:
            pass
        else:
            if stattypes == {"numerical"}:
                nan_mask = ~np.isnan(c1_vals) & ~np.isnan(c2_vals)
                # Store c1, c2 regression.
                r_items_1 = lin_reg(c1_vals[nan_mask], c2_vals[nan_mask])
                results[c1][c2] = r_items_1
                # Slope and intercept have different values in the inverse
                # direction.
                r_items_2 = lin_reg(c2_vals[nan_mask], c1_vals[nan_mask])
                results[c2][c1] = r_items_2
            elif stattypes == {"nominal"}:
                chi_s_items = chi_squared(c1_vals, c2_vals)
                results[c1][c2] = chi_s_items
                results[c2][c1] = chi_s_items
            elif stattypes == {"nominal", "numerical"}:
                if schema[c1].name == "nominal":
                    anova_items = anova(df, c1, c2)
                else:
                    anova_items = anova(df, c2, c1)
                results[c1][c2] = anova_items
                results[c2][c1] = anova_items
            else:
                raise ValueError(stattypes)

    json.dump(results, args.output)


if __name__ == "__main__":
    main()
