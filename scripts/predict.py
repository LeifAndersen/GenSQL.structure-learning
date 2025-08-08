import argparse
import edn_format
import numpy as np
import pandas as pd
import sys
import yaml

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def read_data(path, target):
    """Read data and drop all rows where the target is null."""
    df = pd.read_csv(path)
    return df[~df[target].isnull()]


def error_message_stat_type(stat_type):
    return f"Target type must be numerical or nominal, is {stat_type}"


def impute_missing_features(train_dataset, test_dataset, schema):
    replacements = {}
    for c in train_dataset.columns:
        if schema[c] == "numerical":
            replacements[c] = train_dataset[c].median()
        elif schema[c] == "nominal":
            replacements[c] = train_dataset[c].mode()[0]
        else:
            raise ValueError(error_message_stat_type(schema[c]))
    train_dataset = train_dataset.fillna(replacements)
    test_dataset = test_dataset.fillna(replacements)
    return train_dataset, test_dataset


def one_hot_enconding(train_dataset, test_dataset):
    """Fit SKL's one-hot encoder on all feature data (X), apply,
    and return both as NP arrays.
    """
    encoder = OneHotEncoder(handle_unknown="error")
    train_dataset = train_dataset.astype(str)
    test_dataset = test_dataset.astype(str)
    # We need to fit the encoder on both, the training and the test data
    # to ensure the categoridal encodings in both agrees and is robust to cases
    # where a category doesn't appear in one of them.
    encoder.fit(pd.concat([train_dataset, test_dataset]))
    # Transform training and test data.
    X_transformed = encoder.transform(train_dataset).toarray()
    X_pred_transformed = encoder.transform(test_dataset).toarray()
    return X_transformed, X_pred_transformed


def recode_categoricals(train_dataset, test_dataset, schema, target):
    """Recode categorical columns using one-hot encoding. Keep remaining columns
    and don't recode the target for prediction."""
    categoricals = [
        c for c in train_dataset.columns if (schema[c] == "nominal") and (c != target)
    ]
    X_transformed, X_pred_transformed = one_hot_enconding(
        train_dataset[categoricals], test_dataset[categoricals]
    )
    # Add new cols to df
    col_names = [
        c for c in train_dataset.columns if (schema[c] == "numerical") and (c != target)
    ]
    for i in range(X_transformed.shape[1]):
        col_name = f"c_{i}"
        train_dataset[col_name] = X_transformed[:, i]
        test_dataset[col_name] = X_pred_transformed[:, i]
        col_names.append(col_name)
    return train_dataset, test_dataset, col_names


def prep_data_for_ml(target, train_dataset, test_dataset, schema):
    """Prepare data for machine learning model."""
    # Impute missing values. We apply median- and mode-based imputation.
    # This ensures that one-hot-encoding doesn't encode a missingness flag which
    # may provide an unfair advantage to real over synthetic data if not
    # explicitly modeled in the latter.
    train_dataset, test_dataset = impute_missing_features(
        train_dataset, test_dataset, schema
    )
    # Recode categoricals.
    train_dataset, test_dataset, col_names = recode_categoricals(
        train_dataset, test_dataset, schema, target
    )
    # Return Numpy arrays.
    return (
        train_dataset[col_names].values,
        train_dataset[target].values,
        test_dataset[col_names].values,
        test_dataset[target].values,
    )


def train_random_forest(X, y, target_type):
    """Train a random forest classifier or regressor with CatBoost."""
    if target_type == "nominal":
        return CatBoostClassifier(random_state=42).fit(X, y)
    elif target_type == "numerical":
        return CatBoostRegressor(random_state=42).fit(X, y)
    else:
        raise ValueError(error_message_stat_type(target_type))


def train_glm(X, y, target_type):
    """Train a generalized linear model (GLM)."""
    if target_type == "nominal":
        return LogisticRegression(random_state=42, max_iter=10000).fit(X, y)
    elif target_type == "numerical":
        return LinearRegression().fit(X, y)
    else:
        raise ValueError(error_message_stat_type(target_type))


def train_ml_model(X, y, target_type, ml_model_name):
    """Train a machine learning model."""
    if ml_model_name == "Random_forest":
        return train_random_forest(X, y, target_type)
    elif ml_model_name == "GLM":
        return train_glm(X, y, target_type)
    else:
        raise ValueError(
            f"ML model name must be in ['Random_forest', 'GLM'] but is {ml_model_name}"
        )


def main():
    description = "Evaluate predictions using a ML model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--schema", type=argparse.FileType("r"), help="Path to CGPM schema."
    )
    parser.add_argument(
        "--training",
        action="append",
        type=str,
        help="Training data set.",
        dest="training",
    )
    parser.add_argument(
        "--test",
        action="append",
        type=str,
        help="Test data set.",
        dest="test",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w+"),
        default=sys.stdout,
        metavar="PATH",
    )
    args = parser.parse_args()
    schema = {
        k: v.name
        for k, v in edn_format.loads(args.schema.read(), write_ply_tables=False).items()
    }

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f.read())
    # Get held-out configuration for evaluation.
    config = params["synthetic_data_evaluation"]

    args = parser.parse_args()
    target = config.get(
        "target", np.random.choice([k for k, v in schema.items() if v != "ignore"])
    )

    results = {
        "target": [],
        "training_data": [],
        "test_data": [],
        "prediction": [],
        "true_value": [],
    }
    for train_dataset_path in args.training:
        for test_dataset_path in args.test:
            # Re-reading here because prep_data_for_ml deletes cols.
            train_dataset = read_data(train_dataset_path, target)
            test_dataset_all = read_data(test_dataset_path, target)
            test_dataset = test_dataset_all.sample(
                config.get("N", len(test_dataset_all)), random_state=int(params["seed"])
            )
            X_train, y_train, X_test, y_test = prep_data_for_ml(
                target, train_dataset, test_dataset, schema
            )
            try:
                ml_model = train_ml_model(
                    X_train, y_train, schema[target], config["predictor"]
                )
                # Need to call NP.array.flatten() here because CatBoost decides to
                # wrap prediction into a separate list.
                results["prediction"].extend((ml_model.predict(X_test).flatten().tolist()))
                results["true_value"].extend(y_test.tolist())

                n_test_datapoints = y_test.shape[0]
                results["target"].extend([target] * n_test_datapoints)
                results["training_data"].extend([train_dataset_path] * n_test_datapoints)
                results["test_data"].extend([test_dataset_path] * n_test_datapoints)
            except Exception as e:
                print(f"Couldn't build model: {e}")

    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
