import os
import glob
import time
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import StratifiedKFold

"""
Benchmark model based on different input data containing
combinations of fingerprints and descriptors.

Time-split evaluation is done by the values given in the Set column
as described in the publication:
https://doi.org/10.1021/acs.chemrestox.9b00338

"""


def evaluate_autogluon_by_set(dataset_name, data_df, label_column, id_columns):
    # Separate train and test data by pre-definded set
    df = data_df.copy()
    train_df = df[df["Set"] == "Train"].copy()
    test_df = df[df["Set"] == "Test"].copy()

    train_df.drop(columns=["Set"], inplace=True)
    test_df.drop(columns=["Set"], inplace=True)
    ignored_columns = id_columns.copy()
    ignored_columns.remove("Set")

    # Train model
    predictor = TabularPredictor(
        label=label_column,
        path=f"./models/ag-binary-{dataset_name}-by-set",
        problem_type="binary",
        eval_metric="roc_auc",
        learner_kwargs={"positive_class": "yes", "ignored_columns": ignored_columns},
    )
    predictor.fit(
        # time_limit=10,
        train_data=train_df,
        presets="best_quality",
        verbosity=0,
    )

    # Evaluate model on test data
    y_pred = predictor.predict_proba(test_df.drop(columns=[label_column]))
    y_true = test_df[label_column]
    perf = predictor.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        auxiliary_metrics=True,
        silent=True,
        # detailed_report=True,
    )
    # Add fold and name to performance dictionary
    perf["dataset"] = dataset_name

    # Show roc auc of each fold and dataset
    score = perf.get("roc_auc")
    print(f"ROC-AUC: {score:.4f}\tdataset: {dataset_name}")

    return perf


def evaluate_autogluon(
    dataset_name, data_df, label_column, id_columns, n_splits=5, random_state=0
):
    exclude_cols = [label_column] + id_columns
    X = data_df.drop(columns=exclude_cols).values
    y = data_df[label_column].values
    outer_cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    nested_scores = []
    for fold, (train_inds, test_inds) in enumerate(outer_cv.split(X, y)):
        train_df = data_df.iloc[train_inds, :]
        test_df = data_df.iloc[test_inds, :]

        # Train model
        predictor = TabularPredictor(
            label=label_column,
            path=f"./models/ag-binary-{dataset_name}-fold-{fold:02}",
            problem_type="binary",
            eval_metric="roc_auc",
            learner_kwargs={"positive_class": "yes", "ignored_columns": id_columns},
        )
        predictor.fit(
            # time_limit=10,
            train_data=train_df,
            presets="best_quality",
            verbosity=0,
        )

        # Evaluate model on test data
        y_pred = predictor.predict_proba(test_df.drop(columns=[label_column]))
        y_true = test_df[label_column]
        perf = predictor.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            auxiliary_metrics=True,
            silent=True,
            # detailed_report=True,
        )
        # Add fold and name to performance dictionary
        perf["fold"] = fold
        perf["dataset"] = dataset_name

        # Show roc auc of each fold and dataset
        score = perf.get("roc_auc")
        print(f"ROC-AUC: {score:.4f}\tfold: {fold}\tdataset: {dataset_name}")

        nested_scores.append(perf)
    return nested_scores


def main():
    label_column = "Photosensitation"
    id_columns = ["Substance", "Canonical_Smiles", "Set"]

    datasets = [fname for fname in glob.glob("./data/pih_*.csv")]
    print("\nInput datasets:")
    print(datasets)

    results = []
    times = []
    for i, dataset_name in enumerate(datasets):
        # Read data
        data_df = TabularDataset(data=dataset_name)
        np.random.seed(0)

        # Get name of the dataset
        base = os.path.basename(dataset_name)
        name = os.path.splitext(base)[0]

        print("Starting:", name, data_df.shape)
        start = time.time()
        # scores = evaluate_autogluon(
        #     name, data_df, label_column, id_columns, n_splits=5
        # )
        # results.extend(scores)
        scores = evaluate_autogluon_by_set(name, data_df, label_column, id_columns)
        results.append(scores)
        elapsed = time.time() - start
        times.append(elapsed)
        print("Done. Elapsed time:", elapsed)

    print("\nResults:")
    print(results)
    # Round results to 4 digits
    results = pd.DataFrame(results).round(4)
    results.to_csv("results/time_split_results.csv", index=False)

    # print("\nAggregated results:")
    # # Calculate mean and std of the model metrics evaluated on different test sets for each dataset
    # results_aggregated = (
    #     results.drop(columns=["fold"]).groupby("dataset").agg(["mean", "std"]).round(4)
    # )
    # # Flatten MultiIndex columns from aggregation
    # results_aggregated.columns = results_aggregated.columns.map("|".join).str.strip("|")
    # print(results_aggregated)
    # results_aggregated.to_csv("results/results_aggregated.csv")

    print("\nModel times:")
    print(times)


if __name__ == "__main__":
    main()
