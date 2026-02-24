"""
This script is used to automatically extract performance metrics from
hyper-parameter sweep that is set up with Weights & Biases
"""
import pandas as pd
import wandb
from tqdm import tqdm
import os
api = wandb.Api()

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

sweep_ids = [
    "fjn8c92u", "eks1ypp2", "5dsaegee", "bmpzw4hy",
    "rtm0pta9", "5vmnwij5", "ywc04ms7", "klmwa77v",
    "qau83ym7", "brk1luoz", "8rzfey4x", "nki85db4",
    "zuswcdye", "rhk2va6i", "o2y8e1hi", "5v6m6eys",
    "yr0x9jqh"
]

save_path = f"/localdisk1/{project_dir}/{project_name}/results/R3_Screening_Performance_with_SMOTE/wandb_results/wandb_runs_summary_multi_view_all_runs.csv"
if not os.path.exists(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

summary_list, config_list, name_list, id_list = [], [], [], []

for sweep_id in sweep_ids:
    print(f"Processing sweep: {sweep_id}")

    # Sort by a summary metric (prefix with 'summary_metrics.')
    # Use '+' for ascending, '-' for descending
    runs = api.runs(
        f"{wandb_username}/{project_name}_v1",
        filters={"sweep": sweep_id},
        per_page=100,
        order="-summary_metrics.dev_auroc"
    )

    i = 0
    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        id_list.append(run.id)
        i +=1

        if i%100 == 0:
            print(f"Processed {i} runs for sweep {sweep_id}...")

            runs_df = pd.DataFrame({
                "summary": summary_list,
                "config": config_list,
                "name": name_list,
                "id": id_list
                })
                
            # Flatten the dictionaries into separate columns for a clean CSV
            config_df = pd.json_normalize(runs_df['config'])
            summary_df = pd.json_normalize(runs_df['summary'])
            final_df = pd.concat([runs_df[['name', 'id']], config_df, summary_df], axis=1)
            
            # Export to CSV
            final_df.to_csv(save_path, index=False)
            
            # # # We only need the best 1000 runs, discard the rest to save memory
            # break

final_df.to_csv(save_path, index=False)        
print("Done logging for all sweeps!")