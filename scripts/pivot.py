#!/usr/bin/env python

import argparse
import sys
import yaml
import pandas as pd
import re
import math
import edn_format
from functools import reduce
from sklearn.utils import shuffle
from typing import List

def pandas_to_gensql_types(type):
  if type == "float64":
    return "numerical" 
  if type == "float32":
    return "numerical" 
  if type == "int64":
    return "numerical" 
  elif type == "object":
    return "nominal"
  else:
    return "ignore"

def schema_from_df(df, munge_keys=False):
  return {munge(str(k)) if munge_keys else k: edn_format.Keyword(pandas_to_gensql_types(v)) for k, v in df.dtypes.astype('str').to_dict().items()} 

def determine_schema_element(item, schema):
  if type(item) is tuple:
    ret = None
    [ret := ret or determine_schema_element(i, schema) for i in item]
    return ret
  else:
    s = schema.get(item, None)
    return edn_format.Keyword(s) if s is not None else None

def filter_frame(df: pd.DataFrame, filter_cutoff: float) -> pd.DataFrame:
  """Filter columns with low information value, as determined by `filter_cutoff`"""
  cols_to_keep = [col for col in df.columns 
                  if len(df[col].value_counts(dropna=False)) > 0 and 
                    df[col].value_counts(normalize=True,dropna=False).iloc[0] < 0.75]
  return df[cols_to_keep]

def pivot(df: pd.DataFrame, index: List[str], categories: List[str]) -> pd.DataFrame:
  """Pivot the dataframe across `column`, leaving the `index` columns unchanged."""
  index = [i for i in index if i in df.columns]
  categories = [i for i in categories if i in df.columns]
  category_values = [col for col in df
                     if col not in index and col not in categories]
  dup_key="dup_" + "key" #(categories[0] if len(categories) > 0 else "None")
  df_dup = df.assign(**{dup_key: df.groupby(index+categories, dropna=False).cumcount()})
  return df_dup.pivot(index=index+[dup_key], columns=categories, values=category_values).reset_index()

def munge(s: str):
  """Munges column names to ones the rest of the pipeline can handle.
  Currently this is a lossy operation."""
  ## TODO, make a smarter munger (maybe parameterize it too)
  s = re.sub(r"[()\"']", "", s)
  return re.sub(r"[^0-9A-Za-z_]", "_", s)

def shrink(df: pd.DataFrame, datakey: str, size: float, seed) -> pd.DataFrame:
  """Subsample the dataframe, but keep `datakey` groups in tact."""
  subjects = shuffle(df[datakey].unique(), random_state=seed)
  selected = subjects[0:math.floor(len(subjects)*size)]
  return df[df[datakey].isin(selected)]

def main():
  parser = argparse.ArgumentParser(description="")
  parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w+"),
    default=sys.stdout,
    metavar="PATH",
    )
  parser.add_argument(
    "--full-output",
    type=argparse.FileType("w+"),
    default=sys.stdout,
    metavar="PATH",
    )
  parser.add_argument(
    "--schema-output",
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
  na_rep = "" if len(nullify) == 0 else list(nullify)[0]
  seed = params.get("seed", None)
  schema = params.get("schema", {}) or {}
  guess_schema = params.get("guess_schema", None)

  df = pd.read_csv(args.data, na_values=nullify) # dtype=str

  if "pivot" in params and params["pivot"] is not None:

    ignore = params["pivot"].get("ignore", []) or []
    df = df.drop(columns=ignore, errors='ignore')

    # Remove low information rows
    filter_cutoff = params["pivot"].get("filter_cutoff", 0.75)
    size = params["pivot"].get("size", 1)
    size_key = params["pivot"].get("size_key", None)
    df = filter_frame(df, filter_cutoff)
    for p in params["pivot"].get("steps", []) or []:
      index = p.get("index", []) or []
      keys = p.get("key", []) or []
      if type(keys) is str:
        keys = [keys]
      if len(keys) > 0:
        df = filter_frame(pivot(df, index, keys), filter_cutoff)

    ## Write the unshrunk pivoted CSV
    df.set_axis([munge(str(c)) for c in df], axis=1).to_csv(args.full_output, index=False, na_rep = na_rep)

    # Shrink the data
    if size and size_key:
      df = shrink(df, size_key, size, seed)

    # Guess the schema if we're doing it in the python step
    guessed_schema = schema_from_df(df, True) if guess_schema else {}

    args.schema_output.write(
      edn_format.dumps({**guessed_schema,
                        **{munge(str(c)): element
                           for c in df if (element := determine_schema_element(c, schema)) is not None}}))
    df.set_axis([munge(str(c)) for c in df], axis=1).to_csv(args.output, index=False, na_rep = na_rep)
  else:
    guessed_schema = schema_from_df(df, False) if guess_schema else {}

    args.schema_output.write(
      edn_format.dumps({**guessed_schema,
                        **{c: edn_format.Keyword(schema[c]) for c in schema}}))
    df.to_csv(args.output, index=False, na_rep=na_rep)
    df.to_csv(args.full_output, index=False, na_rep=na_rep)

if __name__ == "__main__":
  main()
