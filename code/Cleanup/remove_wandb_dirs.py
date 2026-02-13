"""
This screipt recursively searches for and removes all "wandb" directories
starting from a specified (local) root directory.

-- Any directory named "wandb" will be permanently deleted along with all its contents.
-- Use with caution, as this action is irreversible.
"""
import subprocess
import os

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

def remove_wandb_recursive(current_dir = "."):
    try:
        dir_lists = os.listdir(current_dir)
    except Exception as e:
        dir_lists = []

    for x in dir_lists:
        if x == "wandb":
            delete_dir_path = os.path.join(current_dir, x)
            print(f"Deleting directory: {delete_dir_path}")
            subprocess.run(["rm", "-rf", delete_dir_path])

        elif os.path.isdir(os.path.join(current_dir, x)):
            remove_wandb_recursive(os.path.join(current_dir, x))
    return

if __name__ == "__main__":
    current_dir = f"/localdisk1/{project_dir}/{project_name}"
    remove_wandb_recursive(current_dir)