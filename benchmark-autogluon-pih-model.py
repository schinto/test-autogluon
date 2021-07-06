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


def calc_sensitivity(df):
    """df contains the confusion_matrix
    as pandas DataFrame
    """
    tp = df.loc["yes", "yes"]
    fn = df.loc["no", "yes"]
    return tp / (tp + fn)


def calc_specifity(df):
    """df contains the confusion_matrix
    as pandas DataFrame
    """
    fp = df.loc["yes", "no"]
    tn = df.loc["no", "no"]
    return tn / (tn + fp)


def evaluate_autogluon_by_set(
    dataset_name, data_df, label_column, id_columns, time_limit=600
):
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
        time_limit=time_limit,
        train_data=train_df,
        presets="best_quality",
        # presets="high_quality_fast_inference_only_refit",
        verbosity=0,
    )

    # Evaluate model on train data
    train_perf = predictor.evaluate(
        data=train_df,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    train_perf["dataset"] = dataset_name
    train_perf["Set"] = "Train"
    train_perf["sensitivity"] = calc_sensitivity(train_perf["confusion_matrix"])
    train_perf["specifity"] = calc_specifity(train_perf["confusion_matrix"])

    # Evaluate model on test data
    test_perf = predictor.evaluate(
        data=test_df,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    test_perf["dataset"] = dataset_name
    test_perf["Set"] = "Test"
    test_perf["sensitivity"] = calc_sensitivity(test_perf["confusion_matrix"])
    test_perf["specifity"] = calc_specifity(test_perf["confusion_matrix"])

    # Evaluate model on External data
    ext_perf = predictor.evaluate(
        data=ext_df,
        auxiliary_metrics=True,
        silent=True,
        detailed_report=True,
    )
    ext_perf["dataset"] = dataset_name
    ext_perf["Set"] = "External"
    ext_perf["sensitivity"] = calc_sensitivity(ext_perf["confusion_matrix"])
    ext_perf["specifity"] = calc_specifity(ext_perf["confusion_matrix"])

    # Remove undesired performance metrics
    remove_keys = ("confusion_matrix", "classification_report")
    for k in remove_keys:
        train_perf.pop(k, None)
        test_perf.pop(k, None)
        ext_perf.pop(k, None)

    # Show evaluation metric
    metric_name = predictor.eval_metric.name
    test_score = test_perf.get(metric_name)
    train_score = train_perf.get(metric_name)
    ext_score = ext_perf.get(metric_name)
    print(f"Dataset: {dataset_name}")
    print(f"Train    {metric_name}: {train_score:.4f}")
    print(f"Test     {metric_name}: {test_score:.4f}")
    print(f"External {metric_name}: {ext_score:.4f}")

    return [train_perf, test_perf, ext_perf]


def main():
    label_column = "Photosensitation"
    id_columns = ["Substance", "Canonical_Smiles", "Set"]

    datasets = [fname for fname in glob.glob("./data/pih_*.csv")]
    print("\nInput datasets:")
    print(datasets)

    time_limit = 600

    results = []
    times = []
    for i, dataset_name in enumerate(datasets):
        # Read data
        data_df = TabularDataset(data=dataset_name)
        np.random.seed(0)

        # Get basename of the dataset without extension
        base = os.path.basename(dataset_name)
        name = os.path.splitext(base)[0]

        print(f"Starting[{i}]: {name} {data_df.shape}")
        start = time.time()
        scores = evaluate_autogluon_by_set(
            name, data_df, label_column, id_columns, time_limit
        )
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
    results.to_csv("results/pih_time_split_results.csv", index=False)
    print(results)

    print("\nModel times:")
    print(times)


if __name__ == "__main__":
    main()
