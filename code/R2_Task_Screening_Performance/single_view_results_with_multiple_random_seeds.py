import pandas as pd
import numpy as np
import os, sys
import random
import wandb
import regex as re
import ast
import json
import torch

wandb_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_single_view_top_100.csv"
summary_results_path = "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/summary_best_models_per_task_with_random_seeds.csv"

model_name_for_display = {
    "VideoMAE": "VideoMAE",
    "ViViT": "ViViT",
    "TimeSformer": "TimeSformer",
    "VideoPrism": "VideoPrism",
    "VideoMAEv2": "VideoMAEv2",
    "VJEPA2": "VJEPA2",
    "VJEPA2_SSV2": "VJEPA2-SSv2"
}

model_name_from_display = {v: k for k, v in model_name_for_display.items()}

# Generate 30 random seeds for producing confidence intervals
random_seeds = [random.randint(0, 10000) for _ in range(30)]
# print("Random Seeds:", random_seeds)

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

        # Get the hyperparameters of the best model
        params = {
            "task_name": best_row["task_name"],
            "num_epochs": int(best_row["num_epochs"]),
            "batch_size": int(best_row["batch_size"]),
            "hidden_dim": int(best_row["hidden_dim"]),
            "drop_prob": float(best_row["drop_prob"]),
            "optimizer": best_row["optimizer"],
            "learning_rate": float(best_row["learning_rate"]),
            "use_scheduler": best_row["use_scheduler"],
            "scheduler": best_row["scheduler"],
            "model": best_row["model"],
            "pooling": best_row["pooling"],
            "num_views": int(best_row["num_views"]),
            "view_index": int(best_row["view_index"]),
            "seed": best_row["seed"]
        }

        # Run experiments with multiple random seeds, but the same other hyperparameters
        results = []
        for seed in random_seeds:
            params["seed"] = seed
            python_command = "python /localdisk1/PARK/park_video_benchmarking/code/R2_Task_Screening_Performance/train_single_view_for_final_results.py"
            cmd = f"{python_command} {' '.join([f'--{k}={v}' for k, v in params.items()])}"
            print(f"Running command: {cmd}")
            status = os.system(cmd)

            result_file = os.path.join(
                "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/temp_run_logs",
                f"wandb_logs_{params['task_name']}_model_{params['model']}_view{params['view_index']}_seed{seed}.json"
            )
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    log_data = json.load(f)
                    results.append(log_data)
        
        # Save detailed results for each task and best model
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/detailed_results_single_view_{task}_model_{best_model}.csv", index=False)

        n = results_df["test_samples"].mean()
        n_positives = results_df["test_positives"].mean()
        
        summary_row = {
            "Task": " ".join([x.capitalize() for x in task.split("_")]),
            "Best Model": model_name_for_display[best_row["model"]],
            "Mean Accuracy": results_df["test_accuracy"].mean(),
            "Std Accuracy": results_df["test_accuracy"].std(),
            "Mean Sensitivity": results_df["test_recall"].mean(),
            "Std Sensitivity": results_df["test_recall"].std(),
            "Mean Specificity": results_df["test_specificity"].mean(),
            "Std Specificity": results_df["test_specificity"].std(),
            "Mean PPV": results_df["test_precision"].mean(),
            "Std PPV": results_df["test_precision"].std(),
            "Mean NPV": results_df["test_npv"].mean(),
            "Std NPV": results_df["test_npv"].std(),
            "Mean AUC": results_df["test_auroc"].mean(),
            "Std AUC": results_df["test_auroc"].std(),
            "Test Samples": n,
            "Test Positives": n_positives,
        }
        print(summary_row)
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

if __name__ == "__main__":
    # # Stage 1 -- slow as it runs every model for 30 different seeds

    # summary_df = summarize_wandb_results(wandb_results_path)
    # print("Summary of Best Models per Task:")
    # print(summary_df)
    # summary_df.to_csv(summary_results_path, index=False)

    # Stage 2
    summary_df = pd.read_csv(summary_results_path)
    # this will hold our final latex table
    results_df = pd.DataFrame()

    for task in summary_df["Task"]:
        # read back results for 30 random seeds
        model = summary_df[summary_df["Task"]==task]["Best Model"].values[0]
        task_str = task.lower().replace(" ", "_")
        task_result_path = f"/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/detailed_results_single_view_{task_str}_model_{model_name_from_display[model]}.csv"
        df_task = pd.read_csv(task_result_path)

        summary_row = {
            "Task": " ".join([x.capitalize() for x in task.split("_")]),
            "Best Model": model,
            "Accuracy": f"${(df_task['test_accuracy'].mean()*100):.1f} \pm {(1.96*(df_task['test_accuracy'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "Sensitivity": f"${(df_task['test_recall'].mean()*100):.1f} \pm {(1.96*(df_task['test_recall'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "Specificity": f"${(df_task['test_specificity'].mean()*100):.1f} \pm {(1.96*(df_task['test_specificity'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "PPV": f"${(df_task['test_precision'].mean()*100):.1f} \pm {(1.96*(df_task['test_precision'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "NPV": f"${(df_task['test_npv'].mean()*100):.1f} \pm {(1.96*(df_task['test_npv'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "AUC": f"${(df_task['test_auroc'].mean()*100):.1f} \pm {(1.96*(df_task['test_auroc'].std()/np.sqrt(len(df_task)))*100):.1f}$",
            "Test Samples": df_task["test_samples"].mean(),
            "Test Positives": df_task["test_positives"].mean(),
            "N": len(df_task)
        }
        results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)


    print("Final Summary with 95% CIs:")
    results_df.to_csv(summary_results_path.replace(".csv", "_final_with_CI_random_seeds.csv"), index=False)
    print(results_df)

    # Make them Latex Ready
    summary_df = results_df.copy()
    summary_df = summary_df[summary_df["Task"]!="Resting Face"]
    summary_df.drop(columns=["Test Samples", "Test Positives", "N"], inplace=True)
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
    with open(summary_results_path.replace(".csv", "_final_with_CI_random_seeds.tex"), "w") as f:
        f.write(latex_code)