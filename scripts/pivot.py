#!/usr/bin/env python

import argparse
import sys
import yaml
import pandas as pd
import re
import math
from sklearn.utils import shuffle
from typing import List

def filter_frame(df: pd.DataFrame, filter_cutoff: float) -> pd.DataFrame:
  """Filter columns with low information value, as determined by `filter_cutoff`"""
  cols_to_keep = [col for col in df.columns 
                  if len(df[col].value_counts(dropna=False)) > 0 and 
                    df[col].value_counts(normalize=True,dropna=False).iloc[0] < 0.75]
  return df[cols_to_keep]

def pivot(df: pd.DataFrame, index: List[str], column: str) -> pd.DataFrame:
  """Pivot the dataframe across `column`, leaving the `index` columns unchanged."""
  categories = [column]
  category_values = [col for col in df
                     if col not in index and col not in categories]
  dup_key="dup_" + column
  df_dup = df.assign(**{dup_key: df.groupby(index+categories, dropna=False).cumcount()})
  return df_dup.pivot(index=index+[dup_key], columns=categories, values=category_values).reset_index()

def munge(s: str):
  """Munges column names to ones the rest of the pipeline can handle.
  Currently this is a lossy operation."""
  ## TODO, make a smarter munger (maybe parameterize it too)
  s = re.sub(r"[()\"']", "", s)
  return re.sub(r"[^0-9A-Za-z_]", "_", s)

def shrink(df: pd.DataFrame, datakey: str, size: float) -> pd.DataFrame:
  """Subsample the dataframe, but keep `datakey` groups in tact."""
  subjects = shuffle(df[datakey].unique())
  selected = subjects[1:math.floor(len(subjects)*size)]
  return df[df[datakey].isin(selected)]

def write_csv(df: pd.DataFrame, name: str):
  """Unused, makes 1%, 10%, 25%, and 100% cutoffs to a dataframe."""
  helper = lambda df, name: df.set_axis([munge(str(c)) for c in df], axis=1).to_csv(name, index=False)
  helper(df, f"{name}-100.csv")
  helper(shrink(df, 0.25), f"{name}-25.csv")
  helper(shrink(df, 0.10), f"{name}-10.csv")
  helper(shrink(df, 0.01), f"{name}-1.csv")

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
    "--params",
    type=argparse.FileType("r"),
    help="Path to params.yaml",
    )
  args = parser.parse_args()

  params = yaml.safe_load(args.params)
  nullify = set(params.get("nullify", []) or [])
  na_rep = "" if len(nullify) == 0 else nullify[0]

  df = pd.read_csv(args.data, na_values=nullify) # dtype=str
  if "pivot" in params and params["pivot"] is not None:
    filter_cutoff = params["pivot"].get("filter_cutoff", 0.75)
    size = params["pivot"].get("size", 1)
    size_key = params["pivot"].get("size_key", None)
    df = filter_frame(df, filter_cutoff)
    for p in params["pivot"].get("steps", []) or []:
      index = p.get("index", []) or []
      key = p.get("key", None)
      if key:
        df = filter_frame(pivot(df, index, key), filter_cutoff)
    if size_key:
      shrink(df, size_key, size)
    df.set_axis([munge(str(c)) for c in df], axis=1).to_csv(args.output, index=False, na_rep = na_rep)
  else:
    df.to_csv(args.output, index=False, na_rep=na_rep)

if __name__ == "__main__":
  main()
