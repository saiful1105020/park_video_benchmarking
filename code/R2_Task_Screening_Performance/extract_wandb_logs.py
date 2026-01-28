import pandas as pd
import wandb
from tqdm import tqdm
api = wandb.Api()

# /localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_all_runs.csv
# Project is specified by <entity/project-name>
# runs = api.runs("mislam6/park_video_benchmarking_v1", per_page=1000)

sweep_ids = ["i4o2kxbt", "wlhk39r2", "78wphc9r", "hhpm976u", "anpqbigc",
             "8ncgwe7g", "8kymj26f", "2ga080l9", "2s3khxvt", "jkvt5udi",
             "2287v01z", "kdhpwwbt", "9e65jeiq", "0sape03p", "cz5xzmdg",
             "gcl0d9ym", "455hbjvp"]

summary_list, config_list, name_list = [], [], []

for sweep_id in sweep_ids:
    print(f"Processing sweep: {sweep_id}")

    # Sort by a summary metric (prefix with 'summary_metrics.')
    # Use '+' for ascending, '-' for descending
    runs = api.runs(
        "mislam6/park_video_benchmarking_v1",
        filters={"sweep": sweep_id},
        per_page=1000,
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
        i +=1

        if i%1000 == 0:
            print(f"Processed {i} runs for sweep {sweep_id}...")

            runs_df = pd.DataFrame({
                "summary": summary_list,
                "config": config_list,
                "name": name_list
                })
                
            # Flatten the dictionaries into separate columns for a clean CSV
            config_df = pd.json_normalize(runs_df['config'])
            summary_df = pd.json_normalize(runs_df['summary'])
            final_df = pd.concat([runs_df[['name']], config_df, summary_df], axis=1)

            # Export to CSV
            final_df.to_csv("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_all_runs.csv", index=False)
            
            # # We only need the best 1000 runs, discard the rest to save memory
            # break

final_df.to_csv("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_all_runs.csv", index=False)        
print("Done logging for all sweeps!")