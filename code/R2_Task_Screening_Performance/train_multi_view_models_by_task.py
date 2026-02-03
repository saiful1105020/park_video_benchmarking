from ast import Index
import os
import sys
import copy
import click
import wandb
import random
import pandas as pd

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt

# os.makedirs("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/", exist_ok=True)
# sys.stdout.flush()
# sys.stdout=open("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/train_models_by_task_dryrun_temp.log", "wt")

from models import *

# task_name_mapping = {
#     'sustained_phonation_a': ['ahhhh'],
#     'sustained_phonation_e': ['eeeee'],
#     'sustained_phonation_o': ['ooooo'],
#     'facial_expression_disgust': ['disgust'],
#     'facial_expression_smile': ['smile'],
#     'facial_expression_surprise': ['surprise'],
#     'extend_arm': ['extend_arm'],
#     'eye_gaze': ['eye_gaze'],
#     'finger_tapping': ['finger_tapping', 'finger_tapping_left', 'finger_tapping_right'],
#     'flip_palm': ['flip_palm', 'flip_palm_left', 'flip_palm_right'],
#     'head_pose': ['head pose'],
#     'nose_touch': ['nose_touch', 'nose_touch_left', 'nose_touch_right'],
#     'open_fist': ['open_fist', 'open_fist_left', 'open_fist_right'],
#     'pangram_utterance': ['quick_brown_fox'],
#     'resting_face': ['resting_face'],
#     'reverse_count': ['reverse_count'],
#     'tongue_twister': ['tongue_twister'],
#     'resting_tremor': ['resting_tremor'],
#     'free_flow_speech': ['speech']
# }

sys.path.append("/localdisk1/PARK/park_video_benchmarking/code/Utils")
from file_path_labels import *
from get_static_embeddings import *
from calculate_performance_metrics import *

# Setup GPU support
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Device: {device}")

valid_tasks = [x for x in list(task_name_mapping.keys()) if x not in ["free_flow_speech", "resting_tremor"]]
print("Valid tasks are:")
print(valid_tasks)

def construct_dataset_df(config):
    '''
    Based on the configurations (pre-trained model, num views, view index),
    load embeddings, and attach them to PD/Non-PD labels.
    Keep the unique ids
    '''
    # get the locations and PD labels for all files that exist in metadata
    dataset = get_file_paths_and_labels(task_name=config["task_name"])
    print(f"Size of training set: {len(dataset['train'])} videos, {len(dataset['train']['pid'].unique())} participants")
    print(f"Size of validation set: {len(dataset['dev'])} videos, {len(dataset['dev']['pid'].unique())} participants")
    print(f"Size of test set: {len(dataset['test'])} videos, {len(dataset['test']['pid'].unique())} participants")

    print(f"Task name: {config['task_name']}")

    # get the saved embeddings from pre-trained models
    del config["task_name"]
    df_embeddings = get_all_static_embeddings(**config)
    print(f"Number of files with embedding: {len(df_embeddings)}")
    
    # combined embedding and label columns into a single dataframe
    n_pd = 0
    n_non_pd = 0
    n_total = 0

    for fold in ["train", "dev", "test"]:
        dataset[fold] = dataset[fold].set_index("file_name").join(df_embeddings.set_index("file_name"), on="file_name", how="inner")
        dataset[fold] = dataset[fold].reset_index()
        print(f"Number of {fold} files with embedding: {len(dataset[fold])}")

        n_pd += dataset[fold]["label"].sum()
        n_non_pd += len(dataset[fold]) - dataset[fold]["label"].sum()
        n_total += len(dataset[fold])

    print(f"Total number of files with embedding: {n_total}, PD: {n_pd}, Non-PD: {n_non_pd}")
    
    # dictionary of three dataframes: 
    # dataset["train"], dataset["dev"], and dataset["test"]
    return dataset

class TensorDataset(Dataset):
    '''
    Standard dataloader for this specific single-view embeddings.
    '''
    def __init__(self, df):
        self.ids = df["unique_id"]
        self.labels = df["label"]
        self.features = df["features"]
    
    def __getitem__(self, index):
        uid = self.ids.iloc[index]
        x = self.features.iloc[index]
        y = self.labels.iloc[index]
        return uid, x, y
    
    def __len__(self):
        return len(self.labels)
    
class TensorDatasetLR(Dataset):
    '''
    Standard dataloader for this specific single-view embeddings.
    Columns: ['file_name', 'pid', 'task', 'protocol', 'label', 'file_path', 'unique_id', 'features']
    '''
    def __init__(self, df):
        df_both_hands = df[(~(df["file_name"].str.contains("left"))) & (~df["file_name"].str.contains("right"))]
        df_single_hand = df[(df["file_name"].str.contains("left")) | df["file_name"].str.contains("right")]
        # df_single_hand.to_csv("single_hand_files.csv", index=False)

        indexes_to_remove = []

        for idx, row in df_single_hand.iterrows():
            # Skip if this index is already matched
            if idx in indexes_to_remove:
                continue

            if "left" in row["file_name"]:
                corresponding_file = row["file_name"].replace("left", "right")
            elif "right" in row["file_name"]:
                corresponding_file = row["file_name"].replace("right", "left")
            else:
                assert False
            
            matched_rows = df_single_hand[df_single_hand["file_name"]==corresponding_file]
            if len(matched_rows)==1:
                matched_row = matched_rows.iloc[0]
                # average the features
                avg_feature = (row["features"] + matched_row["features"])/2.0
                df_single_hand.at[idx, "features"] = avg_feature
                df_single_hand.at[idx, "file_name"] = row["file_name"].replace("_left", "_merged").replace("_right", "_merged")

                # mark the matched index for deletion later
                matched_index = matched_rows.index[0]
                indexes_to_remove.append(matched_index)

            # otherwise, we just keep the original feature (no averaging)

        df_single_hand = df_single_hand.drop(index=indexes_to_remove).reset_index(drop=True)
        # df_single_hand.to_csv("single_hand_files_after_merging.csv", index=False)

        df = pd.concat([df_both_hands, df_single_hand]).reset_index(drop=True)

        self.ids = df["unique_id"]
        self.labels = df["label"]
        self.features = df["features"]

        # Sanity check
        # df1 = pd.read_csv("single_hand_files.csv")
        # df2 = pd.read_csv("single_hand_files_after_merging.csv")
        # print(f"Number of single-hand files before merging: {len(df1)}")
        # print(f"Number of single-hand files after merging: {len(df2)}")
    
    def __getitem__(self, index):
        uid = self.ids.iloc[index]
        x = self.features.iloc[index]
        y = self.labels.iloc[index]
        return uid, x, y
    
    def __len__(self):
        return len(self.labels)
    
def evaluate(model, data_loader, criterion):
    '''
    Evaluation loop (model is not updated)
    Returns the loss and performance metrics
    '''
    model.eval()

    loss_total = 0
    n_total = 0
    all_preds = []
    all_labels = []

    for idx, batch in enumerate(data_loader):
        ids, features, labels = batch
        n_total += len(labels)
        all_labels.extend(labels)

        labels = labels.float().to(device)
        features = features.to(device)

        with torch.no_grad():
            predicted_probs = model(features)
            l = criterion(predicted_probs.reshape(-1), labels)
            loss_total += l.item()*len(labels)
            all_preds.extend(predicted_probs.to('cpu').numpy())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = loss_total/n_total
    return metrics

def training_loop(train_loader, dev_loader, model, optimizer, scheduler, criterion, all_configs):
    '''
    Control the training process.
    Return the best model and performance metrics on training and validation sets.
    '''
    best_model = copy.deepcopy(model)
    best_dev_loss = np.finfo('float32').max
    loss_vs_iterations = [] # to plot the trend of training loss

    for epoch in tqdm(range(all_configs["num_epochs"])):
        # track training loss for each epoch
        training_loss = 0
        n_total = 0
        for idx, batch in enumerate(train_loader):
            ids, features, labels = batch
            n_total += len(labels)
            
            labels = labels.float().to(device)  #(n, )
            features = features.float().to(device)      #(n, d)

            predicted_probs = model(features)   #(n, 1)
            l = criterion(predicted_probs.reshape(-1), labels)
            
            # propagate gradients
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            training_loss += l.item()*len(labels)
            loss_vs_iterations.append(l.item())

        training_loss = training_loss/n_total   # average training loss
        if all_configs["enable_wandb"]:
                wandb.log({"train_loss": l.item()})
        
        # evaluate performance on the validation set
        dev_metrics = evaluate(model, dev_loader, criterion)
        print(f"After epoch {epoch}, training_loss = {training_loss}, validation loss = {dev_metrics['loss']}")

        if all_configs["use_scheduler"]=="yes":
            if all_configs['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_metrics["loss"])

        if all_configs["detailed_logs"]:
            print("Metrics on vaidation set:")
            print(dev_metrics)

        # update the best model if this epoch improved validation performance (loss)
        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            best_model = copy.deepcopy(model)
            if all_configs["detailed_logs"]:
                print("Model updated")

        print("---"*3)
        # end of one epoch

    # end of training
    # evaluate final performance on the training and validation set
    training_metrics = evaluate(best_model, train_loader, criterion)
    dev_metrics = evaluate(best_model, dev_loader, criterion)
    
    # save training loss curve
    if all_configs["detailed_logs"]:
        plt.plot(np.arange(len(loss_vs_iterations)), loss_vs_iterations)
        plt.savefig("training_loss.png", dpi=300)

    return best_model, training_metrics, dev_metrics

# Comment the click decorators for unit testing
@click.command()
@click.option("--task_name", default="sustained_phonation_a", help="Options: see task lists")
@click.option("--num_epochs", default=82)
@click.option("--batch_size", default=256)
@click.option("--hidden_dim", default=512)
@click.option("--drop_prob", default=0.5)
@click.option("--optimizer",default="AdamW",help="Options: SGD, AdamW")
@click.option("--learning_rate", default=0.00001, help="Learning rate for classifier")
@click.option("--momentum", default=0.9)
@click.option("--use_scheduler", default='no',help="Options: yes, no")
@click.option("--scheduler", default='step',help="Options: step, reduce")
@click.option("--step_size", default=11)
@click.option("--gamma", default=0.8808588244592819)
@click.option("--patience", default=3)
@click.option("--detailed_logs", default=False)
@click.option("--model", default="TimeSformer", help="Options: ViViT, VideoMAE, TimeSformer")
@click.option("--pooling", default="max", help="Options: max, mean")
@click.option("--num_views", default=4)
@click.option("--view_index", default=-1)
@click.option("--seed", default=42)
@click.option("--enable_wandb", default=True)
def main(**cfg):
    assert cfg["task_name"] in valid_tasks, f"Invalid task name. Valid options are: {valid_tasks}"
    assert cfg["model"] in model_embedding_paths.keys(), f"Invalid model name. Valid options are: {list(model_embedding_paths.keys())}"

    # need to setup wandb and hyper-parameter tuning
    ENABLE_WANDB = cfg["enable_wandb"]
    if ENABLE_WANDB:
        wandb.init(project="park_video_benchmarking_v1", config=cfg)
    '''
    Ensure reproducibility of randomness
    '''
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # these are configs for model training
    all_configs = cfg

    # these are configs for embedding extraction
    # datasets: all_tasks, facial_expressions, free_flow_speech
    config = {
        "task_name": cfg["task_name"],
        "model": cfg["model"],
        "pooling": cfg["pooling"],
        "num_views": cfg["num_views"],
        "view_index": cfg["view_index"]
    }

    # setup dataset, data loader, model, optimizer, scheduler, loss function

    dataset_df = construct_dataset_df(config)
    
    # print(dataset_df.keys())    # dict_keys(['dev', 'test', 'train'])
    # print(dataset_df["test"].columns)   # Index(['file_name', 'pid', 'task', 'protocol', 'label', 'file_path', 'unique_id', 'features'],
    # print(dataset_df["dev"].iloc[0])
    # print(dataset_df["train"].iloc[101]["features"].dtype) # torch.float32
    # print(dataset_df["train"].iloc[101]["features"].shape) # torch.Size([4, 768])
    # assert False

    if cfg["task_name"] in ["finger_tapping", "flip_palm", "nose_touch", "open_fist"]:
        train_dataset = TensorDatasetLR(df=dataset_df["train"])
        dev_dataset = TensorDatasetLR(df=dataset_df["dev"])
        test_dataset = TensorDatasetLR(df=dataset_df["test"])
    else:
        train_dataset = TensorDataset(df=dataset_df["train"])
        dev_dataset = TensorDataset(df=dataset_df["dev"])
        test_dataset = TensorDataset(df=dataset_df["test"])

    # print(len(train_dataset))   # 1641
    # uid, x, y = train_dataset[0]
    # print(x.shape)  # torch.Size([4, 768])
    # print(uid, y)   # sustained_phonation_a_269 0
    # assert False

    train_loader = DataLoader(dataset=train_dataset, batch_size=all_configs["batch_size"], shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=all_configs["batch_size"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=all_configs["batch_size"])
    
    _, x, _ = train_dataset[0]
    print(f"Feature shape: {x.shape}")  # torch.Size([n_views, n_features])
    n_features = x.shape[1]
    print(f"Number of features per view: {n_features}")
    model = MultiViewLinearProbe(n_views = cfg["num_views"], n_features=n_features, hidden_dim=all_configs["hidden_dim"], drop_prob=all_configs["drop_prob"])
    model.to(device)

    criterion = nn.BCELoss()

    if all_configs["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=all_configs['learning_rate'])
    elif all_configs["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=all_configs['learning_rate'], momentum=all_configs['momentum'])
    elif all_configs["optimizer"]=="RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=all_configs['learning_rate'], momentum=all_configs['momentum'])
    else:
        raise ValueError("Invalid optimizer")
    
    scheduler = None
    if all_configs["use_scheduler"]=="yes":
        if all_configs['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=all_configs['step_size'], gamma=all_configs['gamma'])
        elif all_configs['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=all_configs['gamma'], patience = all_configs['patience'])
        else:
            raise ValueError("Invalid scheduler")

    # train the model    
    trained_model, training_metrics, dev_metrics = training_loop(
        train_loader=train_loader, dev_loader=dev_loader, model=model, 
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        all_configs=all_configs)
    
    wandb_logs = {}
    print("="*10)
    print("Training metrics")
    print(training_metrics)
    wandb_logs["train_loss"] = training_metrics["loss"]
    print("Validation metrics")
    print(dev_metrics)
    wandb_logs["dev_loss"] = dev_metrics["loss"]
    wandb_logs["dev_auroc"] = dev_metrics["auroc"]
    wandb_logs["dev_accuracy"] = dev_metrics["accuracy"]
    wandb_logs["dev_f1_score"] = dev_metrics["f1_score"]
    
    # evaluate on the test set
    test_metrics = evaluate(model=trained_model, data_loader=test_loader, criterion=criterion)
    print("Test metrics")
    print(test_metrics)
    wandb_logs["test_accuracy"] = test_metrics["accuracy"]
    wandb_logs["test_precision"] = test_metrics["precision"]
    wandb_logs["test_recall"] = test_metrics["recall"]
    wandb_logs["test_auroc"] = test_metrics["auroc"]
    wandb_logs["test_f1_score"] = test_metrics["f1_score"]
    wandb_logs["test_specificity"] = test_metrics["specificity"]
    wandb_logs["test_npv"] = test_metrics["NPV"]
    wandb_logs["test_samples"] = test_metrics["confusion_matrix"]["tp"] + test_metrics["confusion_matrix"]["tn"] + test_metrics["confusion_matrix"]["fp"] + test_metrics["confusion_matrix"]["fn"]
    wandb_logs["test_positives"] = test_metrics["confusion_matrix"]["tp"] + test_metrics["confusion_matrix"]["fn"]

    if ENABLE_WANDB:
        wandb.log(wandb_logs)
    # save the model
    # after we find out the best configuration

def unit_test():
    for task in valid_tasks:
        for model_name in model_embedding_paths_multiview[4].keys():
            print(f"Training for task: {task}, model: {model_name}")
            config = {
                "task_name": task,
                "num_epochs": 20,
                "batch_size": 256,
                "hidden_dim": 512,
                "drop_prob": 0.5,
                "optimizer":"AdamW",
                "learning_rate":0.00001,
                "momentum":0.9,
                "use_scheduler":"no",
                "scheduler":"step",
                "step_size":11,
                "gamma":0.8808588244592819,
                "patience":3,
                "detailed_logs":False,
                "model": model_name,
                "pooling":"mean",
                "num_views":4,
                "view_index":-1,
                "seed":42,
                "enable_wandb":False
            }
            main(**config)
            print("SUCCESS")
            print("=========================================")

if __name__ == "__main__":
    # unit_test()
    main()

    # End of program
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # os.rename("/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/train_models_by_task_dryrun_temp.log", "/localdisk1/PARK/park_video_benchmarking/results/R2_Task_Screening_Performance/train_models_by_task_dryrun.log")
        
    
