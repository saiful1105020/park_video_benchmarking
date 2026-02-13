'''
This script removes specific large patch files named "diff.patch" from all runs
in a specified Weights & Biases (W&B) project to free up storage space.

Make sure to clean up since free wandb storage is limited.

You can modify this to clean up other unneeded files as well.
'''
import wandb
import pandas as pd
from tqdm import tqdm

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

# Initialize the W&B API
# If, for the first time, you need to login, run `wandb login` in your terminal
api = wandb.Api()

# Define your entity, project, and run ID
entity = wandb_username # your W&B username or team name
project = f"{project_name}_v1" # your W&B project name

# Get runs from the project
runs = api.runs(f"{entity}/{project}")

i = 0

for run in tqdm(runs):
    run_id = run.id    
    try:
        # Get the run object
        run = api.run(f"{entity}/{project}/{run_id}")
        
        file_name = "diff.patch"
        files = run.files()
        
        found = False
        for file in files:
            
            if file.name == file_name:
                print(f"Deleting file: {file.name} from run {run.id}")
                file.delete()
                print(f"{file.name} deleted successfully.")
                found = True
                break
        
        if not found:
            print(f"{file_name} not found in run {run.id}.")

    except Exception as e:
        print(f"An error occurred while processing run {run_id}: {e}")

    i +=1
    if i % 100 == 0:
        print(f"Processed {i} runs out of {len(runs)}...")

