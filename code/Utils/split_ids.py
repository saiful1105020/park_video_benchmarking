import os
import sys

# copy dev and test IDs from the fusion project
DEV_IDS_PATH = "/localdisk1/PARK/park_vlm/Dataset/dev_participant_ids.txt"
TEST_IDS_PATH = "/localdisk1/PARK/park_vlm/Dataset/test_participant_ids.txt"

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