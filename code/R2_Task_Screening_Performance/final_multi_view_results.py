import pandas as pd
import numpy as np
import os, sys
import random
import wandb
import regex as re
import ast
import json

wandb_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_multi_view_top_100.csv"
summary_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/summary_best_models_per_task_multi_view.csv"
latex_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/latex/summary_best_models_per_task_multi_view_final_with_CI.tex"

model_name_for_display = {
    "VideoMAE": "VideoMAE",
    "ViViT": "ViViT",
    "TimeSformer": "TimeSformer",
    "VideoPrism": "VideoPrism",
    "VideoMAEv2": "VideoMAEv2",
    "VJEPA2": "VJEPA2",
    "VJEPA2_SSV2": "VJEPA2-SSv2"
}

# def extract_dicts_from_log(log_data, key_name):
#     # Pattern to find everything between curly braces (non-nested)
#     matches = re.findall(r"\{(?:[^{}]|(?R))*\}", log_data, re.DOTALL)
#     match = matches[-1]  # Get the last match
    
#     data = eval(match, {"np": np, "__builtins__": {}})
#     return data

# def load_wandb_log(run_id):
#     api = wandb.Api()
#     run = api.run(f"mislam6/park_video_benchmarking_v1/{run_id}")
#     file = run.file("output.log")
#     file.download("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/temp_run_logs", replace=True)

#     with open("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/temp_run_logs/output.log", "r") as f:
#         log_data = f.read()
#         metrics_dict = extract_dicts_from_log(log_data, "metrics")
#     return metrics_dict

# Task - Best Model - Accuracy - Sensitivity - Specificity - PPV - NPV - AUC
def summarize_wandb_results(wandb_results_path):
    df = pd.read_csv(wandb_results_path)
    
    summary_df = pd.DataFrame()
    summary_rows = []
    task_names = df["task_name"].unique()
    
    for task in task_names:
        print(f"Processing task: {task}")
        df_task = df[df["task_name"] == task]

        # sort based on dev_auroc and get the best model
        df_task = df_task.sort_values(by="dev_auroc", ascending=False)
        idx_best = df_task["dev_auroc"].idxmax()
        best_row = df_task.loc[idx_best]
        best_model = best_row["model"]

        # # select top-30 rows of the best model only
        # df_task = df_task[df_task["model"] == best_model].head(30)
        # print(df_task.columns)

        # # Extract additional metrics that are missing from this CSV
        # # e.g., Specificity, NPV, #Test_Samples
        # run_id = best_row["id"]
        # extra_metrics = load_wandb_log(run_id)
        # cm = extra_metrics.get("confusion_matrix")
        # n = cm["tn"] + cm["tp"] + cm["fn"] + cm["fp"]
        # n_positives = cm["tp"] + cm["fn"]
        
        summary_row = {
            "Task": " ".join([x.capitalize() for x in task.split("_")]),
            "Best Model": model_name_for_display[best_row["model"]],
            "Accuracy": best_row["test_accuracy"],
            "Sensitivity": best_row["test_recall"],
            "Specificity": best_row["test_specificity"],
            "PPV": best_row["test_precision"],
            "NPV": best_row["test_npv"],
            "AUC": best_row["test_auroc"],
            "Test Samples": best_row["test_samples"],
            "Test Positives": best_row["test_positives"],
        }

        print(summary_row)

        # for i, r in df_task.iterrows():
        #     # Get run id
        #     run_id = r["id"]
        #     # Load the wandb log file for this run to extract missing metrics
        #     metrics = load_wandb_log(run_id)

        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

if __name__ == "__main__":
    # Stage 1
    summary_df = summarize_wandb_results(wandb_results_path)
    print("Summary of Best Models per Task:")
    print(summary_df)
    summary_df.to_csv(summary_results_path, index=False)

    # Stage 2
    summary_df = pd.read_csv(summary_results_path)

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

            if metric in ["Test Samples", "Test Positives"]:
                summary_df.at[i, metric] = f"{int(p)}"
            else:
                summary_df.at[i, metric] = f"${(p*100):.1f} \pm {(1.96*se*100):.1f}$"

    print("Final Summary with 95% CIs:")
    summary_df.to_csv(summary_results_path.replace(".csv", "_final_with_CI.csv"), index=False)
    print(summary_df)

    # Make them Latex Ready
    summary_df = summary_df[summary_df["Task"]!="Resting Face"]
    summary_df.drop(columns=["Test Samples", "Test Positives"], inplace=True)
    new_order = ['Task', 'Best Model', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    summary_df = summary_df.reindex(columns=new_order)
    summary_df.sort_values(by=["Task"], inplace=True)
    
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
    with open(latex_path, "w") as f:
        f.write(latex_code)