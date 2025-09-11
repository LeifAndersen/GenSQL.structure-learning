#!/usr/bin/env python

import argparse
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def predict(df_raw):
    categories = [i for i in df_raw if df_raw[i].dtype == "object"]
    nominal = [i for i in df_raw if df_raw[i].dtype != "object"]

    #enc = OneHotEncoder(sparse_output=False)
    #enc.fit(df_raw[categories])
    #df_temp = enc.transform(df_raw[categories])
    #df = pd.concat([df_raw[nominal], pd.DataFrame(columns=enc.get_feature_names_out(),
    #                                              data=df_temp)],
    #                axis=1)
    df = pd.get_dummies(df_raw).dropna()

    clf = IsolationForest(n_estimators=10, warm_start=True)
    clf.fit(df)
    lof = LocalOutlierFactor()
    km = KMeans()
    km.fit(df)


    return pd.concat([df, 
                      pd.DataFrame(km.labels_, columns=["Cluster"]),
                      pd.DataFrame(clf.predict(df), columns=["IsolationForest"]),
                      pd.DataFrame(lof.fit_predict(df), columns=["LocalOutliers"])
                      ],
                     axis=1)

def main():
    description = "Create simple predictions using a ML models"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w+"),
        default=sys.stdout,
        metavar="PATH",
    )
    parser.add_argument("--data", type=argparse.FileType("r"), help="Path to raw CSV.")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.data)
    df = predict(df_raw)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
