import os
import copy
import pickle
import re
import math
import json
import wandb
import random
import click
import imblearn
import scipy
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import subprocess as sp

import baal.bayesian.dropout as mcdropout
from baal.modelwrapper import ModelWrapper

from pandas import DataFrame
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTEN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

sys.path.append("/localdisk1/PARK/park_video_benchmarking/code/Utils")
from file_path_labels import *
from get_static_embeddings import *
from calculate_performance_metrics import *
from models import *

wandb_temp_path = "/localdisk1/PARK/park_video_benchmarking/results/R5_MultiTask/wandb_results/temp_run_logs"
MODEL_CONFIG_PATH = "/localdisk1/PARK/park_video_benchmarking/results/R5_MultiTask/model_config.txt"
os.makedirs(os.path.basename(wandb_temp_path), exist_ok=True)
os.makedirs(os.path.basename(MODEL_CONFIG_PATH), exist_ok=True)

NUM_TASKS = 0 #just initiate here, later updated based on config

valid_tasks = list(task_name_mapping.keys())
TASK_SUBSETS = {
    0: ['pangram_utterance', 'facial_expression_smile', 'finger_tapping'],
    1: ['pangram_utterance', 'facial_expression_disgust', 'flip_palm', 'eye_gaze'],
    2: ['pangram_utterance', 'facial_expression_disgust', 'flip_palm']
}

best_models_per_task = {
    "pangram_utterance": "VideoPrism",
    "facial_expression_smile": "VideoPrism",
    "finger_tapping": "TimeSformer",
    "facial_expression_disgust": "VJEPA2",
    "flip_palm": "VJEPA2_SSV2",
    "eye_gaze": "VideoPrism"
}

pool_settings_per_task = {
    "pangram_utterance": "mean",
    "facial_expression_smile": "mean",
    "finger_tapping": "max",
    "facial_expression_disgust": "mean",
    "flip_palm": "mean",
    "eye_gaze": "mean"
}

'''
Find the GPU that has max free space
'''
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#1. Load dev and test sets (participant ids)
with open("/localdisk1/PARK/park_video_benchmarking/code/Utils/dev_participant_ids.txt") as f:
    ids = f.readlines()
    dev_ids = set([x.strip() for x in ids])

with open("/localdisk1/PARK/park_video_benchmarking/code/Utils/test_participant_ids.txt") as f:
    ids = f.readlines()
    test_ids = set([x.strip() for x in ids])
	
print(f"Number of patients in the dev and test set: {len(dev_ids)}, {len(test_ids)}")

'''
Parse date from filenames. 
Some examples:
    2022-03-24T13%3A32%3A36.977Z_NIHNT179KNNF4_finger_tapping_left.mp4 -- 2022-03-24
    2021-08-30T20%3A00%3A03.162Z_ZTi20lXEMSdqXLxtnTotwoyADq03_finger_tapping_left.mp4 -- 2021-08-30
    NIHYM875FLXFF-finger_tapping-2021-03-17T18-13-01-902Z-.mp4 -- 2021-03-17
    2019-10-21T22-16-00-772Z35-finger_tapping.mp4 -- 2019-10-21
'''
def parse_date(name:str):
    match = re.search(r"\d{4}-\d{2}-\d{2}", name)
    date = match.group()
    return date

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
        dataset[fold]["id"] = dataset[fold]["pid"]
        dataset[fold]["date"] = dataset[fold]["file_name"].apply(parse_date)
        dataset[fold]["id_date"] = dataset[fold]["id"]+"#"+dataset[fold]["date"]

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
        self.ids = np.asarray(df["id"])
        self.labels = np.asarray(df["label"])
        self.features = np.asarray(df["features"])
        self.dates = np.asarray(df["date"])
        self.id_dates = np.asarray(df["id_date"])
    
    def __getitem__(self, index):
        uid = self.ids[index]
        x = self.features[index]
        y = self.labels[index]
        id_date = self.id_dates[index]
        return uid, x, y, id_date
    
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

        self.ids = np.asarray(df["id"])
        self.labels = np.asarray(df["label"])
        self.features = np.asarray(df["features"])
        self.dates = np.asarray(df["date"])
        self.id_dates = np.asarray(df["id_date"])

        # Sanity check
        # df1 = pd.read_csv("single_hand_files.csv")
        # df2 = pd.read_csv("single_hand_files_after_merging.csv")
        # print(f"Number of single-hand files before merging: {len(df1)}")
        # print(f"Number of single-hand files after merging: {len(df2)}")
    
    def __getitem__(self, index):
        uid = self.ids[index]
        x = self.features[index]
        y = self.labels[index]
        id_date = self.id_dates[index]
        return uid, x, y, id_date
    
    def __len__(self):
        return len(self.labels)
    
class FusionDataset(Dataset):
    def __init__(self, df_list):
        merged_df = None
        for i in range(len(df_list)):
            df_temp = pd.DataFrame()
            df_temp["id"] = df_list[i].ids
            df_temp["date"] = df_list[i].dates
            df_temp["id_date"] = df_list[i].id_dates
            df_temp["labels"] = df_list[i].labels
            df_temp["features"] = df_list[i].features
            df_temp = df_temp.drop_duplicates(subset=["id_date"])
            # print(f"Number of samples in task {i} before merging: {len(df_temp)}")
            
            if merged_df is None:
                merged_df = df_temp
            else:
                merged_df = pd.merge(merged_df, df_temp, on=["id_date"], how="inner")
                merged_df = merged_df.rename(columns = {"features_x": "features"})
                merged_df = merged_df.rename(columns = {"features_y": f"features_{i}"})
                merged_df = merged_df.drop(columns=[x for x in merged_df.columns if "_x" in x])
                merged_df = merged_df.rename(columns = {"id_y": "id", "date_y": "date", "labels_y": "labels"})
                
        merged_df = merged_df.rename(columns = {"features": f"features_0"})
        print(f"Number of samples in task {i} after merging: {len(merged_df)}")
        
        self.merged_df = merged_df
        self.ids = np.asarray(merged_df["id"])
        self.labels = np.asarray(merged_df["labels"])
        self.dates = np.asarray(merged_df["date"])
        self.id_dates = np.asarray(merged_df["id_date"])
        self.n_tasks = len(df_list)

        self.features = []
        for i in range(len(df_list)):
            f = torch.Tensor(np.asarray(merged_df[f"features_{i}"].tolist()))
            self.features.append(f)
    
    def __getitem__(self, index):
        features = []
        for i in range(self.n_tasks):
            features.append(self.features[i][index])

        return features, self.labels[index]

    def __len__(self):
        return len(self.labels)

'''
Given a dataframe, perform oversampling
input_df: must contain columns features_0, features_1, ..., features_(N-1), and label
output_df: oversamples the minority class and returns in similar format
other columns (i.e., id, filename, etc.) will be removed.
'''
def concat_features(row):
    return np.concatenate([row[f"features_{i}"] for i in range(NUM_TASKS)])

def concat_finger_features(row):
    return np.concatenate([row[f"features_right"], row[f"features_left"]])

def oversample(input_df, sampler):
    feature_shapes = [input_df.iloc[0][f"features_{i}"].shape[0] for i in range(NUM_TASKS)]
    input_df["concat_features"] = input_df.apply(concat_features, axis=1)
    features = input_df.loc[:, "concat_features"]
    labels = input_df.loc[:,"label"]

    X = np.asarray([features.iloc[i] for i in range(len(features))])
    Y = np.asarray([labels.iloc[i] for i in range(len(labels))])

    X, Y = sampler.fit_resample(X, Y)
    output_data = []
    for (x,y) in zip(X,Y):
        data = {}
        start_index = 0
        for i in range(len(feature_shapes)):
            end_index = start_index + feature_shapes[i]
            data[f"features_{i}"] = x[start_index:end_index]
            start_index = end_index

        data["label"] = y
        output_data.append(data)

    output_df = pd.DataFrame.from_dict(output_data)
    return output_df

'''
ML baselines using pytorch + BAAL
'''
class ANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(in_features=n_features, out_features=(int)(n_features/2), bias=True)
        self.drop1 = mcdropout.Dropout(p = drop_prob)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=1,bias=True)
        self.drop2 = mcdropout.Dropout(p = drop_prob)
        self.hidden_activation = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.hidden_activation(self.fc1(x))
        x1 = self.drop1(x1)
        y = self.fc2(x1)
        y = self.drop2(y)
        y = self.sig(y)
        return y

'''
ML baselines using pytorch + BAAL
'''
class ShallowANN(nn.Module):
    def __init__(self, n_features, drop_prob):
        super(ShallowANN, self).__init__()
        self.fc = nn.Linear(in_features=n_features, out_features=1,bias=True)
        self.drop = mcdropout.Dropout(p = drop_prob)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y = self.fc(x)
        y = self.drop(y)
        y = self.sig(y)
        return y

'''
Final predictor
Contains two modules:
    1. a custom cross-attention module
    2. prediction network
'''
class CrossAttention(nn.Module):
    def __init__(self, input_dim, query_dim, drop_prob):
        super(CrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.drop_prob = drop_prob
        
        self.form_query = torch.nn.Linear(input_dim, query_dim)
        self.form_key = torch.nn.Linear(input_dim, query_dim)
        self.form_value = torch.nn.Linear(input_dim, query_dim)

        self.drop = mcdropout.Dropout(p = drop_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.final_layer = torch.nn.Linear((NUM_TASKS-1) * self.input_dim, self.input_dim)

    def forward(self, features):
        queries = []
        keys = []
        values = []
        for i in range(NUM_TASKS):
            q = self.form_query(features[i])
            q = self.drop(q)

            key = self.form_key(features[i])
            key = self.drop(q)

            val = self.form_value(features[i])
            val = self.drop(val)

            queries.append(q)
            keys.append(key)
            values.append(val)

        queries = torch.stack(queries) #(N, n, d)
        queries = queries.transpose(0,1) #(n, N, d)
        keys = torch.stack(keys) #(N, n, d)
        keys_T = keys.transpose(0,1).transpose(-1,-2) #(n, d, N)
        values = torch.stack(values) #(N, n, d)
        values = values.transpose(0,1) #(n, N, d)

        scores = torch.matmul(queries, keys_T) #(n, N, N)
        scores = self.softmax(scores)
        
        zs = torch.matmul(scores, values) #(n, N, d)
        z = zs.reshape((-1, NUM_TASKS*self.query_dim)) #(n, N*d)
        return z
    
class HybridFusionNetworkEndToEnd(nn.Module):
    def __init__(self, feature_shapes, config):
        super(HybridFusionNetworkEndToEnd, self).__init__()
        self.unimodal_predictive_models = nn.ModuleList()
        for i in range(NUM_TASKS):
            model = SingleViewLinearProbe(feature_shapes[i], config["unimodal_hidden_dim"], config["dropout_prob"])
            self.unimodal_predictive_models.append(model)

        self.hidden_dim = config["hidden_dim"]
        self.query_dim = config["query_dim"]
        self.last_hidden_dim = config["last_hidden_dim"]
        self.drop_prob = config["dropout_prob"]
        
        '''
        input: features_i is of shape (feature_shapes[i]); y_pred_score_i
        total input size: feature_shapes[i]+1
        '''
        self.intra_linear = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = CrossAttention(self.hidden_dim, self.query_dim, self.drop_prob) #shared weights

        for i in range(NUM_TASKS):
            linear_layer = nn.Linear(in_features=feature_shapes[i], out_features=self.hidden_dim, bias=True)
            self.intra_linear.append(linear_layer)

        self.lin1 = nn.Linear(in_features=((NUM_TASKS*self.query_dim)+NUM_TASKS), out_features=self.last_hidden_dim)
        self.fc = nn.Linear(in_features=self.last_hidden_dim, out_features=1)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.intra_linear_dropout = mcdropout.Dropout(p = self.drop_prob)
        self.lin1_dropout = mcdropout.Dropout(p = self.drop_prob)
    
    def forward(self, inputs):
        predicted_scores = []

        for i in range(NUM_TASKS):
            inputs[i] = inputs[i].to(device)
            unimodal_outputs = self.unimodal_predictive_models[i](inputs[i]).squeeze(dim=-1) #(n,)
            predicted_scores.append(unimodal_outputs)
        
        hiddens = []
        for i in range(NUM_TASKS):
            hidden_representation = self.relu(self.intra_linear[i](inputs[i])) #projection: (n, d_{x_i}) -> (n,d)
            hidden_representation = self.intra_linear_dropout(hidden_representation)
            hidden_representation = self.relu(hidden_representation)
            hidden_representation = self.layer_norm(hidden_representation)
            hiddens.append(hidden_representation)

        pred_scores = torch.stack(predicted_scores).transpose(0,1).to(device) #(n, N)
        
        context = self.cross_attention(torch.cat([torch.unsqueeze(hiddens[k],0) for k in range(NUM_TASKS)])) #(n, d_q)
        outputs = torch.cat((context, pred_scores),dim=-1) #(n, N+d_q)
        outputs = self.lin1(outputs) #(n, last_hidden_dim)
        outputs = self.lin1_dropout(outputs) 
        logits = self.fc(outputs) #(n,1)
        probs = self.sigmoid(logits) #(n,1)
        return probs

'''
Evaluate performance on validation/test set.
Returns all the metrics defined above and the loss.
'''
def expected_calibration_error(y, y_pred_scores, num_buckets=20):
    y_pred_scores = np.asarray(y_pred_scores).flatten()
    
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, num_buckets + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.maximum(y_pred_scores, 1.0-y_pred_scores)

    # get predictions from confidences (positional in this case)
    predicted_label = (y_pred_scores>=0.5)
    
    # get a boolean list of correct/false predictions
    accuracies = (predicted_label==y)

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece.item()

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

'''
Given labels and prediction scores, make a comprehensive evaluation. 
i.e., threshold = 0.5 means prediction>0.5 will be considered as positive
'''
def compute_metrics(y_true, y_pred_scores, threshold = 0.5):
    labels = np.asarray(y_true).reshape(-1)
    pred_scores = np.asarray(y_pred_scores).reshape(-1)
    preds = (pred_scores >= threshold)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['average_precision'] = average_precision_score(labels, pred_scores)
    metrics['auroc'] = roc_auc_score(labels, pred_scores)
    metrics['f1_score'] = f1_score(labels, preds)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["confusion_matrix"] = {"tn":tn, "fp":fp, "fn":fn, "tp":tp}
    metrics["weighted_accuracy"] = (safe_divide(tp, tp + fp) + safe_divide(tn, tn + fn)) / 2.0

    '''
    True positive rate or recall or sensitivity: probability of identifying a positive case 
    (often called the power of a test)
    '''
    metrics['TPR'] = metrics['recall'] = metrics['sensitivity'] = recall_score(labels, preds)
    
    '''
    False positive rate: probability of falsely identifying someone as positive, who is actually negative
    '''
    metrics['FPR'] = safe_divide(fp, fp+tn)
    
    '''
    Positive Predictive Value: probability that a patient with a positive test result 
    actually has the disease
    '''
    metrics['PPV'] = metrics['precision'] = precision_score(labels, preds)
    
    '''
    Negative predictive value: probability that a patient with a negative test result 
    actually does not have the disease
    '''
    metrics['NPV'] = safe_divide(tn, tn+fn)
    
    '''
    True negative rate or specificity: probability of a negative test result, 
    conditioned on the individual truly being negative
    '''
    metrics['TNR'] = metrics['specificity'] = safe_divide(tn,(tn+fp))

    '''
    Brier score
    '''
    metrics['BS'] = brier_score_loss(labels, pred_scores)

    '''
    Expected Calibration Error
    '''
    metrics['ECE'] = expected_calibration_error(labels, pred_scores)
    
    return metrics

'''
Main evaluation loop to test the fusion model
'''
def evaluate_fusion_model(fusion_model, dataloader, config, split="dev"):
    fusion_model.eval()
    z_critical = scipy.stats.t.ppf(q=0.975, df = config["num_trials"]-1)

    all_labels = [] #true labels
    all_final_predictions = [] #fusion predictions
    uncertain_indices = [] #indices where the 95% CI contains 0.5
    loss = 0 #average loss
    n_samples = 0 #number of examples in the dataloader

    criterion = torch.nn.BCELoss() #loss function
    fusion_model.eval()
    wrapped_fusion_model = ModelWrapper(fusion_model, criterion)
    
    for idx, batch in enumerate(dataloader):
        x = [[] for i in range(NUM_TASKS)] #[x0, x1, ..., xn]
        (x, y) = batch
        y = y.float().to(device)
        all_labels.extend(y.to('cpu').numpy())
        
        #forward pass
        with torch.no_grad():
            final_pred_scores = wrapped_fusion_model.predict_on_batch((x), iterations=config["num_trials"])
            standard_error = (z_critical*final_pred_scores.std(dim=-1).reshape(-1))/math.sqrt(len(final_pred_scores))
            final_pred_scores = final_pred_scores.mean(dim=-1).reshape(-1)
            index_mask = (final_pred_scores-standard_error<=0.50) & (final_pred_scores+standard_error>=0.50)
            n = final_pred_scores.shape[0]
            loss += criterion(final_pred_scores.reshape(-1), y)*n
            n_samples+=n
        
        all_final_predictions.extend(final_pred_scores.cpu().numpy())
        uncertain_indices.extend(index_mask.cpu().numpy())

    #evaluate
    uncertain_indices = np.asarray(uncertain_indices).flatten()
    all_labels = np.asarray(all_labels).flatten()
    all_final_predictions = np.asarray(all_final_predictions).flatten()
    
    if split=="test":
        coverage = (len(all_labels) - uncertain_indices.sum())/len(all_labels)
        all_labels = all_labels[~uncertain_indices]
        all_final_predictions = all_final_predictions[~uncertain_indices]

    metrics = compute_metrics(all_labels, all_final_predictions)
    metrics["loss"] = loss.to('cpu').item() / n_samples
    if split=="test":
        metrics['coverage'] = coverage

    return metrics

@click.command()
@click.option("--learning_rate", default=0.001, help="Learning rate for classifier")
@click.option("--dropout_prob", default=0.30)
@click.option("--unimodal_dropout_prob", default=0.30)
@click.option("--num_buckets", default=20, help="Options: 5, 10, 20, 50, 100")
@click.option("--num_trials", default=30, help="Options: 100-1000")
@click.option("--uncertainty_weight", default=0)
@click.option("--train_random_noise", default="no", help="Options: yes, no")
@click.option("--validation_random_noise", default="no", help="Options: yes, no")
@click.option("--increase_variance",default="no", help="Options: yes, no")
@click.option("--temperature", default=0.05, help="Float between 0 and 1")
@click.option("--noise_variance",default=0.01,help="Float between 0 and 1")
@click.option("--task_subset_choice", default=2, help="3 possible choices.")
@click.option("--seed", default=423, help="Seed for random")
@click.option("--batch_size",default=1024)
@click.option("--num_epochs",default=164)
@click.option("--unimodal_hidden_dim", default=256)
@click.option("--hidden_dim", default=512)
@click.option("--query_dim", default=64)
@click.option("--last_hidden_dim", default=128)
@click.option("--optimizer",default="SGD",help="Options: SGD, AdamW, RMSprop")
@click.option("--beta1",default=0.9)
@click.option("--beta2",default=0.999)
@click.option("--weight_decay",default=0.0001)
@click.option("--momentum",default=0.9)
@click.option("--use_scheduler",default='no',help="Options: yes, no")
@click.option("--scheduler",default='reduce',help="Options: step, reduce")
@click.option("--step_size",default=5)
@click.option("--gamma",default=0.1)
@click.option("--patience",default=5)
@click.option("--enable_wandb", default=True, help="Whether to log the results to wandb")
def main(**cfg):
    global NUM_TASKS

    if cfg["enable_wandb"]:
        wandb.init(project="park_video_benchmarking_v1", config=cfg)

    # '''
    # save the configurations obtained from wandb (or command line) into the model config file
    # '''
    # with open(MODEL_CONFIG_PATH,"w") as f:
    #     f.write(json.dumps(cfg))

    #reproducibility control
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"]) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # update global variable for number of tasks
    selected_tasks = TASK_SUBSETS[cfg["task_subset_choice"]]
    NUM_TASKS = len(selected_tasks)
    
    # data processing is a bit pain.
    # first generate datasets that automatically splits between train/dev/test, and then merge them together for the fusion model.
    processed_datasets = {"train":[], "dev":[], "test":[]}

    for i in range(NUM_TASKS):
        # extract features, labels, ids, columns, row_ids for all task datasets
        # using single view settings
        task_name = selected_tasks[i]
        config = {
            "task_name": task_name,
            "model": best_models_per_task[task_name],
            "pooling": pool_settings_per_task[task_name],
            "num_views": 1,
            "view_index": 0
        }

        # utilize dataset classes that works for all tasks
        dataset_dict = construct_dataset_df(config)
        if task_name in ["finger_tapping", "flip_palm", "nose_touch", "open_fist"]:
            train_dataset = TensorDatasetLR(df=dataset_dict["train"])
            dev_dataset = TensorDatasetLR(df=dataset_dict["dev"])
            test_dataset = TensorDatasetLR(df=dataset_dict["test"])
        else:
            train_dataset = TensorDataset(df=dataset_dict["train"])
            dev_dataset = TensorDataset(df=dataset_dict["dev"])
            test_dataset = TensorDataset(df=dataset_dict["test"])

        # merge the datasets together for later processing
        processed_datasets["train"].append(train_dataset)
        processed_datasets["dev"].append(dev_dataset)
        processed_datasets["test"].append(test_dataset)

    # Fusion dataset merges features based on the unique id defined by the "id_date" column, which is a combination of participant id and date.
    train_dataset = FusionDataset(processed_datasets["train"])
    dev_dataset = FusionDataset(processed_datasets["dev"])
    test_dataset = FusionDataset(processed_datasets["test"])

    print("=="*20)
    for fold in ["train", "dev", "test"]:
        print(f"Number of samples in {fold} set after merging: {len(train_dataset)}, {len(dev_dataset)}, {len(test_dataset)}")
    print("=="*20)

    # Standard dataloader
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size = cfg['batch_size'])

    # we will need feature_shapes to design the fusion model architecture, so we can just get it from the dataset
    features, label = train_dataset[0]
    feature_shapes = [features[i].shape[0] for i in range(NUM_TASKS)]
    print(f"Feature shapes for each task: {feature_shapes}")
    # print(f"Sample label: {label}")

    fusion_model = HybridFusionNetworkEndToEnd(feature_shapes, cfg)
    fusion_model = fusion_model.to(device)
    
    criterion = nn.BCELoss()
    if cfg["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(fusion_model.parameters(),lr=cfg['learning_rate'],betas=(cfg['beta1'],cfg['beta2']),weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(fusion_model.parameters(),lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    elif cfg["optimizer"]=="RMSprop":
        optimizer = torch.optim.RMSprop(fusion_model.parameters(), lr=cfg['learning_rate'], momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Invalid optimizer")

    if cfg["use_scheduler"]=="yes":
        if cfg['scheduler']=="step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
        elif cfg['scheduler']=="reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['gamma'], patience = cfg['patience'])
        else:
            raise ValueError("Invalid scheduler")

    best_model = copy.deepcopy(fusion_model)
    best_dev_loss = np.finfo('float32').max
    best_dev_accuracy = 0
    best_dev_balanced_accuracy = 0
    best_dev_auroc = 0
    best_dev_f1 = 0

    for epoch in tqdm(range(cfg['num_epochs'])):
        all_labels = []
        
        for idx, batch in enumerate(train_loader):
            x = [[] for i in range(NUM_TASKS)]
            (x, y) = batch
            y = y.float().to(device)
            all_labels.extend(y.to('cpu').numpy())
            
            #forward pass
            optimizer.zero_grad()
            final_predictions = fusion_model(x)
            l = criterion(final_predictions.reshape(-1),y)
            l.backward()
            optimizer.step()

            if cfg["enable_wandb"]:
                wandb.log({"train_loss": l.to('cpu').item()})

        #eval on dev set
        dev_metrics = evaluate_fusion_model(fusion_model, dev_loader, cfg)
        dev_loss = dev_metrics["loss"]
        dev_accuracy = dev_metrics["accuracy"]
        dev_balanced_accuracy = dev_metrics["weighted_accuracy"]
        dev_auroc = dev_metrics["auroc"]
        dev_f1 = dev_metrics["f1_score"]
        dev_ece = dev_metrics["ECE"]
        #print(f"Epoch {epoch}: dev accuracy: {dev_metrics['accuracy']}")

        if cfg['use_scheduler']=="yes":
            if cfg['scheduler']=='step':
                scheduler.step()
            else:
                scheduler.step(dev_loss)

        if dev_loss<best_dev_loss:
            best_model = copy.deepcopy(fusion_model)

            best_dev_loss = dev_loss
            best_dev_accuracy = dev_accuracy
            best_dev_balanced_accuracy = dev_balanced_accuracy
            best_dev_auroc = dev_auroc
            best_dev_f1 = dev_f1
            best_dev_ece = dev_ece

    test_metrics = evaluate_fusion_model(best_model, test_loader, cfg, split="test")
    if cfg["enable_wandb"]:
        wandb.log(test_metrics)
        wandb.log({"dev_accuracy":best_dev_accuracy, "dev_balanced_accuracy":best_dev_balanced_accuracy, "dev_loss":best_dev_loss, "dev_auroc":best_dev_auroc, "dev_f1":best_dev_f1, "dev_ece":best_dev_ece})
    print(test_metrics)

    # # '''
    # # Save best model
    # # '''
    # torch.save(best_model.to('cpu').state_dict(),MODEL_PATH)

    # loaded_model = HybridFusionNetworkWithUncertainty(feature_shapes, cfg)
    # loaded_model.load_state_dict(torch.load(MODEL_PATH))
    # loaded_model = loaded_model.to(device)
    # print("="*20)
    # print(evaluate_fusion_model(loaded_model, test_loader, prediction_models, cfg, split="test"))

if __name__ == "__main__":
    main()
