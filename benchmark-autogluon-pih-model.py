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


def sensitivity(df):
    """df contains the confusion_matrix
    as pandas DataFrame
    """
    tp = df.loc["yes", "yes"]
    fn = df.loc["no", "yes"]
    return tp / (tp + fn)


def specifity(df):
    """df contains the confusion_matrix
    as pandas DataFrame
    """
    fp = df.loc["yes", "no"]
    tn = df.loc["no", "no"]
    return tn / (tn + fp)


def evaluate_autogluon_by_set(dataset_name, data_df, label_column, id_columns):
    """Evaluate with pre-defined split from Set column"""
    # Separate train, test and external data using Set
    df = data_df.copy()
    train_df = df[df["Set"] == "Train"].copy()
    test_df = df[df["Set"] == "Test"].copy()
    ext_df = df[df["Set"] == "Ext"].copy()

    train_df.drop(columns=["Set"], inplace=True)
    test_df.drop(columns=["Set"], inplace=True)
    ext_df.drop(columns=["Set"], inplace=True)
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

    # Evaluate model on train data
    y_pred = predictor.predict_proba(train_df.drop(columns=[label_column]))
    y_true = train_df[label_column]
    train_perf = predictor.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    # Add name to performance dictionary
    train_perf["dataset"] = dataset_name
    train_perf["Set"] = "Train"
    train_perf["sensitivity"] = sensitivity(train_perf["confusion_matrix"])
    train_perf["specifity"] = specifity(train_perf["confusion_matrix"])

    # Evaluate model on test data
    y_pred = predictor.predict_proba(test_df.drop(columns=[label_column]))
    y_true = test_df[label_column]
    test_perf = predictor.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    # Add fold and name to performance dictionary
    test_perf["dataset"] = dataset_name
    test_perf["Set"] = "Test"
    test_perf["sensitivity"] = sensitivity(test_perf["confusion_matrix"])
    test_perf["specifity"] = specifity(test_perf["confusion_matrix"])

    # Evaluate model on External data
    y_pred = predictor.predict_proba(ext_df.drop(columns=[label_column]))
    y_true = ext_df[label_column]
    ext_perf = predictor.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    # Add fold and name to performance dictionary
    ext_perf["dataset"] = dataset_name
    ext_perf["Set"] = "External"
    ext_perf["sensitivity"] = sensitivity(ext_perf["confusion_matrix"])
    ext_perf["specifity"] = specifity(ext_perf["confusion_matrix"])

    # Remove undesired performance metrics
    remove_keys = ("confusion_matrix", "classification_report")
    for k in remove_keys:
        train_perf.pop(k, None)
        test_perf.pop(k, None)
        ext_perf.pop(k, None)

    # Show roc auc
    test_score = test_perf.get("roc_auc")
    train_score = train_perf.get("roc_auc")
    ext_score = ext_perf.get("roc_auc")
    print(f"Dataset: {dataset_name}")
    print(f"Train    ROC-AUC: {train_score:.4f}")
    print(f"Test     ROC-AUC: {test_score:.4f}")
    print(f"External ROC-AUC: {ext_score:.4f}")

    return [train_perf, test_perf, ext_perf]


def evaluate_autogluon(
    dataset_name, data_df, label_column, id_columns, n_splits=5, random_state=0
):
    """Random split evaluation"""
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

    datasets = [fname for fname in glob.glob("./data/pih_flatring.csv")]
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
        scores = evaluate_autogluon_by_set(name, data_df, label_column, id_columns)
        results.extend(scores)
        elapsed = time.time() - start
        times.append(elapsed)
        print("Done. Elapsed time:", elapsed)

    print("\nResults:")
    # print(results)
    # Round results to 4 digits
    results = pd.DataFrame(results).round(4)
    reorder_cols = [
        "dataset",
        "Set",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specifity",
        "mcc",
        "f1",
        "precision",
        "recall",
    ]
    rename_cols = {
        "dataset": "Features",
        "Set": "Set",
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "sensitivity": "Sensitivity",
        "specifity": "Specifity",
        "mcc": "MCC",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }

    results = results[reorder_cols].rename(columns=rename_cols)
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
