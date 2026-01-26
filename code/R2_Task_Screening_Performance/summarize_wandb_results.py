import pandas as pd
import numpy as np
import os, sys
import random

wandb_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_export_2026-01-22T16_36_52.255-05_00.csv"
summary_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/summary_best_models_per_task.csv"

model_name_for_display = {
    "VideoMAE": "VideoMAE",
    "ViViT": "ViViT",
    "TimeSformer": "TimeSformer",
    "VideoPrism": "VideoPrism",
    "VideoMAEv2": "VideoMAEv2",
    "VJEPA2": "VJEPA2",
    "VJEPA2_SSV2": "VJEPA2-SSv2"
}

# Task - Best Model - Accuracy - Sensitivity - Specificity - PPV - NPV - AUC
def summarize_wandb_results(wandb_results_path):
    df = pd.read_csv(wandb_results_path)
    
    summary_rows = []
    task_names = df["task_name"].unique()
    
    for task in task_names:
        df_task = df[df["task_name"] == task]
        idx_best = df_task["dev_auroc"].idxmax()
        best_row = df_task.loc[idx_best]
        
        # some metrics are not stored in the CSV, need to explore the log file from the name of the run
        summary_row = {
            "Task": " ".join([x.capitalize() for x in task.split("_")]),
            "Best Model": model_name_for_display[best_row["model"]],
            "Accuracy": best_row["test_accuracy"],
            "Sensitivity": best_row["test_recall"],
            "Specificity": best_row["Name"],
            "PPV": best_row["test_precision"],
            "NPV": best_row["Name"],
            "AUC": best_row["test_auroc"],
            "Test Samples": best_row["Name"],
        }
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

if __name__ == "__main__":
    # Stage 1
    summary_df = summarize_wandb_results(wandb_results_path)
    print("Summary of Best Models per Task:")
    print(summary_df)
    summary_df.to_csv(summary_results_path, index=False)

    # assert False
    # At this step, some manual extraction is needed from the wandb logs to fill in missing metrics
    # e.g., Specificity, NPV, #Test_Samples
    # Then we can work on generating confidence intervals for these metrics
    # Eventually, generate a latex table for the final report

    # Stage 2
    summary_df = pd.read_csv(summary_results_path)

    # These are dummy replacements for missing values for demonstration purposes
    summary_df["Specificity"] = summary_df["Specificity"].apply(lambda x: random.uniform(0.7, 0.9))
    summary_df["NPV"] = summary_df["NPV"].apply(lambda x: random.uniform(0.4, 0.9))
    summary_df["Test Samples"] = summary_df["Test Samples"].apply(lambda x: random.randint(50, 200))
    # These dummy values should be replaced with actual extracted values from wandb logs

    for i, r in summary_df.iterrows():
        n = r["Test Samples"]

        for metric in summary_df.columns:
            if metric == "Test Samples" or metric == "Task" or metric == "Best Model":
                continue

            p = float(r[metric])
            # 95% CI using normal approximation
            se = np.sqrt(p * (1 - p) / n)
            ci_lower = p - 1.96 * se
            ci_upper = p + 1.96 * se
            summary_df.at[i, metric] = f"${(p*100):.1f} \pm {(1.96*se*100):.1f}$"

    print("Final Summary with 95% CIs:")
    print(summary_df)
    summary_df.drop(columns=["Test Samples"], inplace=True)
    summary_df.sort_values(by=["Task"], inplace=True)
    summary_df.to_csv(summary_results_path.replace(".csv", "_with_CI.csv"), index=False)
    latex_code = summary_df.to_latex(
        index=False, 
        label="tab:race_dist",
        escape=False,
        bold_rows=True,
        multicolumn_format='c'
    )

    # Manually replace the environment tags
    latex_code = latex_code.replace("\\begin{table}", "\\begin{table*}")
    latex_code = latex_code.replace("\\end{table}", "\\end{table*}")

    # Save to your path
    with open(summary_results_path.replace(".csv", "_with_CI.tex"), "w") as f:
        f.write(latex_code)