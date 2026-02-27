import os
import sys

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

with open("../protocol_name.txt", "r") as f:
    protocol_name = f.read().strip()

# copy dev and test IDs from the fusion project
DEV_IDS_PATH = f"/localdisk1/{project_dir}/{project_name}/code/Utils/dev_participant_ids.txt"
TEST_IDS_PATH = f"/localdisk1/{project_dir}/{project_name}/code/Utils/test_participant_ids.txt"

def get_dev_ids():
    with open(DEV_IDS_PATH, "r") as f:
        dev_ids = [x.strip() for x in f.readlines()]
    return dev_ids

def get_test_ids():
    with open(TEST_IDS_PATH, "r") as f:
        test_ids = [x.strip() for x in f.readlines()]
    return test_ids

if __name__ == "__main__":
    print(get_dev_ids())
    print(get_test_ids())