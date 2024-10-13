import pandas as pd


def print_performance(metrics, split):
    print(f"\n{split.capitalize()} Set Performance")
    print("=" * 40)
    for key, value in metrics.items():
        if ("ROC_AUC" in key) and ("Overall_ROC_AUC" not in key):
            for i, auc in value.items():
                print(f"{key}_{i}: {auc:.4f}" if auc != "N/A" else f"{key}_{i}: {auc}")
        elif "ROC_Curves" in key:
            continue
        elif "Confusion_Matrix" in key:
            print("\nConfusion Matrix:\n", value)
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\n" + "=" * 40 + "\n")


def save_metrics_to_csv(metrics, save_path):
    results_df = pd.DataFrame()
    for key, value in metrics.items():
        if not isinstance(value, dict):
            results_df[key] = [value]
        else:
            for sub_key, sub_value in value.items():
                results_df[f"{key}_{sub_key}"] = [sub_value]

    results_df = results_df.T.reset_index()
    results_df.columns = ["Metric", "Value"]
    csv_path = save_path + "_" + "performance_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Performance metrics saved to {csv_path}")
