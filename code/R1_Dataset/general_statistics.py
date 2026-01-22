"""
This script will be used to describe the dataset, including basic 
statistics such as number of patients, number of videos, distribution of labels, etc.
"""
import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.stdout.flush()
sys.stdout=open("/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/general_statistics_temp.log", "wt")

PLOT_SAVE_DIR = '/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/plots'
CSV_SAVE_DIR = '/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/csv'
LATEX_SAVE_DIR = '/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/latex_tables'

if os.path.exists(PLOT_SAVE_DIR) is False:
    os.makedirs(PLOT_SAVE_DIR)

if os.path.exists(CSV_SAVE_DIR) is False:
    os.makedirs(CSV_SAVE_DIR)

if os.path.exists(LATEX_SAVE_DIR) is False:
    os.makedirs(LATEX_SAVE_DIR)

min_age = 21
max_age = 90

# ['ahhhh' 'disgust' 'eeeee' 'extend_arm' 'eye_gaze' 'finger_tapping_left'
#  'finger_tapping_right' 'flip_palm_left' 'flip_palm_right' 'head pose'
#  'nose_touch_left' 'nose_touch_right' 'ooooo' 'open_fist_left'
#  'open_fist_right' 'quick_brown_fox' 'resting_face' 'reverse_count'
#  'smile' 'surprise' 'tongue_twister' 'finger_tapping' 'flip_palm'
#  'nose_touch' 'open_fist' 'resting_tremor' 'speech']

# Had to exclude 'resting_tremor' and 'speech' tasks due to insufficient data
# task_name_mapping = {
#     'Sustained Phonation (A)': ['ahhhh'],
#     'Sustained Phonation (E)': ['eeeee'],
#     'Sustained Phonation (O)': ['ooooo'],
#     'Facial Expression (Disgust)': ['disgust'],
#     'Facial Expression (Smile)': ['smile'],
#     'Facial Expression (Surprise)': ['surprise'],
#     'Extend Arm': ['extend_arm'],
#     'Eye Gaze': ['eye_gaze'],
#     'Finger Tapping': ['finger_tapping', 'finger_tapping_left', 'finger_tapping_right'],
#     'Flip Palm': ['flip_palm', 'flip_palm_left', 'flip_palm_right'],
#     'Head Pose': ['head pose'],
#     'Nose Touch': ['nose_touch', 'nose_touch_left', 'nose_touch_right'],
#     'Open Fist': ['open_fist', 'open_fist_left', 'open_fist_right'],
#     'Pangram Utterance': ['quick_brown_fox'],
#     'Resting Face': ['resting_face'],
#     'Reverse Count': ['reverse_count'],
#     'Tongue Twister': ['tongue_twister'],
#     'Resting Tremor': ['resting_tremor'],
#     'Free Flow Speech': ['speech']
# }

task_name_mapping = {
    'Sustained Phonation (A)': ['ahhhh'],
    'Sustained Phonation (E)': ['eeeee'],
    'Sustained Phonation (O)': ['ooooo'],
    'Facial Expression (Disgust)': ['disgust'],
    'Facial Expression (Smile)': ['smile'],
    'Facial Expression (Surprise)': ['surprise'],
    'Extend Arm': ['extend_arm'],
    'Eye Gaze': ['eye_gaze'],
    'Finger Tapping': ['finger_tapping', 'finger_tapping_left', 'finger_tapping_right'],
    'Flip Palm': ['flip_palm', 'flip_palm_left', 'flip_palm_right'],
    'Head Pose': ['head pose'],
    'Nose Touch': ['nose_touch', 'nose_touch_left', 'nose_touch_right'],
    'Open Fist': ['open_fist', 'open_fist_left', 'open_fist_right'],
    'Pangram Utterance': ['quick_brown_fox'],
    'Resting Face': ['resting_face'],
    'Reverse Count': ['reverse_count'],
    'Tongue Twister': ['tongue_twister'],
}

sex_mapping = {
    'Female': ['female', 'Female'],
    'Male': ['male', 'Male'],
    'Other': ['Nonbinary'],
    'Unknown': ['Prefer not to respond']
}

race_mapping = {
    'white': ['White', "['White']", "['Asian', 'White']", 'white', 'nativeAmerican, white', 'white,', 'asian,white,', 'white,black,', 'nativeAmerican,white,black,', 'white,race', "['White', 'Asian']"],
    'Black or African American': ['Black or African American', "['Black or African American']", 'black', 'black,race', 'black,'],
    'American Indian or Alaska Native': ['American Indian or Alaska Native', "['American Indian or Alaska Native']",'nativeAmerican', 'nativeAmerican,race'],
    'Native Hawaiian or Other Pacific Islander': ['nativePacific,'],
    'Asian': ['Asian', "['Asian']", 'asian', 'asian,', 'asian,race'],
    'Other': ["['Other']", 'other,race', 'on,', 'other,'],
    'Unknown': ['Prefer not to answer', "['Prefer not to respond']",'[]']
}

def newline():
    print("----------------------------------------")
    print("\n")

def filename_to_date(filename):
    # Examples:

    # input: 2022-03-17T20%3A07%3A18.321Z_NIHFT628PHTAY_ahhhh.mp4
    # output: 2022-03-17

    # input: NIHFT628PHTAY-ahhhh-2020-02-28T19-56-33-169Z-.mp4
    # output: 2020-02-28

    # input: 2021-12-13T19%3A23%3A44.103Z_ihPGkVbviPdKgnZcTHaLFwjNRNH2_surprise.mp4
    # output: 2021-12-13

    # input: 2021-10-04T16%3A05%3A46.663Z_2D4ZYpey5zYz1eHE1U2bY537bsA3_open_fist_left.mp4
    # output: 2021-10-04

    # input: 2022-04-26T22%3A21%3A28.028Z_2FY04QenjbN6QhIMK29gkZfLBhw1_quick_brown_fox.mp4
    # output: 2022-04-26

    # input: 2024-05-30T00%3A42%3A19.464Z_x1ZRr7UucUUP87nREjTLpkqAurI2_open_fist_right.mp4
    # output: 2024-05-30

    # input: 2023-05-31T18%3A44%3A36.261Z_h5A3y2K3k9Q77ltRFJVQ6F49GZf2_quick_brown_fox.mp4
    # output: 2023-05-31

    # input: 2017-11-21T04-14-19-518Z16-finger_tapping.mp4
    # output: 2017-11-21

    # input: 2022-04-25T21%3A19%3A03.020Z_gLLoSHz3bcWvjtnqUikyBTaOAak1_resting_face.mp4
    # output: 2022-04-25

    match = re.search(r"(\d{4}-\d{2}-\d{2})T", filename)
    date = match.group(1)
    return date

def filename_to_time(filename):
    # Examples:

    # input: 2022-03-17T20%3A07%3A18.321Z_NIHFT628PHTAY_ahhhh.mp4
    # output: 20-07-18-321

    # input: NIHFT628PHTAY-ahhhh-2020-02-28T19-56-33-169Z-.mp4
    # output: 19-56-33-169

    # input: 2021-12-13T19%3A23%3A44.103Z_ihPGkVbviPdKgnZcTHaLFwjNRNH2_surprise.mp4
    # output: 19-23-44-103

    # input: 2021-10-04T16%3A05%3A46.663Z_2D4ZYpey5zYz1eHE1U2bY537bsA3_open_fist_left.mp4
    # output: 16-05-46-663

    # input: 2022-04-26T22%3A21%3A28.028Z_2FY04QenjbN6QhIMK29gkZfLBhw1_quick_brown_fox.mp4
    # output: 22-21-28-028

    # input: 2024-05-30T00%3A42%3A19.464Z_x1ZRr7UucUUP87nREjTLpkqAurI2_open_fist_right.mp4
    # output: 00-42-19-464

    # input: 2023-05-31T18%3A44%3A36.261Z_h5A3y2K3k9Q77ltRFJVQ6F49GZf2_quick_brown_fox.mp4
    # output: 18-44-36-261

    # input: 2017-11-21T04-14-19-518Z16-finger_tapping.mp4
    # output: 04-14-19-518

    # input: 2022-04-25T21%3A19%3A03.020Z_gLLoSHz3bcWvjtnqUikyBTaOAak1_resting_face.mp4
    # output: 21-19-03-020

    pattern = re.compile(
        r"T(\d{2})(?:%3A|-)(\d{2})(?:%3A|-)(\d{2})(?:\.|-)(\d{3})"
    )
    
    m = pattern.search(filename)
    if not m:
        return None
    return "-".join(m.groups())

def clean_data():
    data_path = "/localdisk1/PARK/park_video_benchmarking/data/metadata/all_file_user_metadata.csv"
    data_df = pd.read_csv(data_path)

    # Columns: 
    # 'Filename', 'Protocol', 'Participant_ID', 'Task', 'Duration', 'FPS',
    # 'Frame_Height', 'Frame_Width', 'gender', 'age', 'race', 'ethnicity',
    # 'pd', 'dob', 'time_mdsupdrs'
    
    print(f"[METADATA] Number of videos before filtering: {data_df.shape[0]}") #39106
    newline()

    # Exclude data with no PD label
    pd_labels = data_df["pd"].unique()
    print(f"PD labels found in dataset: {pd_labels}")
    data_df = data_df[~data_df["pd"].isna()]
    print(f"[METADATA] Number of videos after filtering no PD label: {data_df.shape[0]}")
    newline()

    # Remove data with suspicious age values
    data_df = data_df[data_df['age'].isna() | ((data_df['age']>=min_age) & (data_df['age']<=max_age))]
    print(f"[METADATA] Number of videos after filtering invalide age: {data_df.shape[0]}")
    newline()

    # Exclude tasks that are not relevant
    #  'ahhhh' 'disgust' 'eeeee' 'extend_arm' 'eye_gaze' 'finger_tapping_left'
    #  'finger_tapping_right' 'flip_palm_left' 'flip_palm_right' 'head pose'
    #  'nose_touch_left' 'nose_touch_right' 'ooooo' 'open_fist_left'
    #  'open_fist_right' 'quick_brown_fox' 'resting_face' 'reverse_count'
    #  'smile' 'surprise' 'tongue_twister' 'finger_tapping' 'flip_palm'
    #  'nose_touch' 'open_fist' 'task1' 'resting_tremor' 'speech' 'task16'
    #  'task14' 'task15'
    data_df = data_df[~data_df["Task"].isin([
        'task1', 'task14', 'task15', 'task16', 'resting_tremor', 'speech'
        ])]
    task_names = data_df["Task"].unique()
    print(f"Task names found in dataset: {task_names}")
    print(f"[METADATA] Number of videos after filtering tasks: {data_df.shape[0]}")
    newline()

    # Exclude data with non-existing videos
    existing_videos = set(os.listdir("/localdisk1/PARK/park_video_benchmarking/data/videos/raw_videos/"))
    filenames_in_metadata = set(data_df["Filename"].tolist())
    print(f"Number of existing videos in folder: {len(existing_videos)}")
    print(f"Number of filenames in metadata: {len(filenames_in_metadata)}")
    missing_videos = filenames_in_metadata - existing_videos
    print(f"Number of missing videos: {len(missing_videos)}")
    data_df = data_df[~data_df["Filename"].isin(missing_videos)]
    print(f"[METADATA] Number of videos after filtering non-existing videos: {data_df.shape[0]}")
    newline()

    # Correct Participant_ID for ParkTest
    print(f"Number of unique Participant_IDs before correction: {data_df['Participant_ID'].nunique()}")
    data_df.loc[data_df['Protocol']=='ParkTest', 'Participant_ID'] = data_df.loc[data_df['Protocol']=='ParkTest', 'Participant_ID'].apply(lambda x: x.split('-')[-1])
    print(f"Number of unique Participant_IDs after correction: {data_df['Participant_ID'].nunique()}")

    # Extract date and time from filename and add as new columns
    data_df['Date'] = data_df['Filename'].apply(lambda x: filename_to_date(x))
    data_df['Time'] = data_df['Filename'].apply(lambda x: filename_to_time(x))

    # Standardize Sex column
    # print(data_df["gender"].unique()) # 'female' 'male' 'Male' 'Female' 'Prefer not to respond' 'Nonbinary'
    data_df.loc[data_df['gender'].isna(), 'gender'] = 'Unknown'
    for x in sex_mapping.keys():
        data_df.loc[data_df['gender'].isin(sex_mapping[x]), 'gender'] = x
    print(data_df['gender'].unique())
    newline()

    # Standardize Race column
    # print(data_df["race"].unique())
    data_df.loc[data_df['race'].isna(), 'race'] = 'Unknown'
    for x in race_mapping.keys():
        data_df.loc[data_df['race'].isin(race_mapping[x]), 'race'] = x
    print(data_df['race'].unique())
    newline()

    # Add PD stage column for clinically diagnosed subjects
    pd_stage_df = pd.read_csv("/localdisk1/PARK/park_video_benchmarking/data/metadata/df_stage.csv")
    pd_stage_df.rename(columns={'id':'Participant_ID', 'mdsupdrshoehnyahrstagescor':'PD_Stage'}, inplace=True)
    data_df = data_df.merge(pd_stage_df[['Participant_ID', 'PD_Stage']], on='Participant_ID', how='left')

    # Save cleaned metadata
    cleaned_metadata_path = "/localdisk1/PARK/park_video_benchmarking/data/metadata/cleaned_file_user_metadata.csv"
    data_df.to_csv(cleaned_metadata_path, index=False)
    print(f"Cleaned metadata saved to: {cleaned_metadata_path}")
    newline()
    return data_df

def plot_age_distribution(data_df):
    # print(df["pd"].unique())  # ['no' 'yes']
    df = data_df.drop_duplicates(subset=['Participant_ID'])

    # 1. Prepare the subgroups
    df_with_pd = df[df['pd'] == 'yes'].copy()
    df_with_pd['Group'] = 'With PD'

    df_without_pd = df[df['pd'] == 'no'].copy()
    df_without_pd['Group'] = 'Without PD'

    # 2. Prepare the "Total" group by duplicating the whole dataset
    df_total = df.copy()
    df_total['Group'] = 'Total'

    # 3. Concatenate the groups into one DataFrame for plotting
    plot_df = pd.concat([df_with_pd, df_without_pd, df_total]).reset_index(drop=True)

    # Define your bin edges
    bin_edges = list(range(20, max_age+1, 10)) # [20, 30, 40, ..., 100]

    # Calculate the midpoints for the labels
    # Formula: (edge_start + edge_end) / 2
    bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

    # Create the tick labels (e.g., "20-30", "30-40")
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=plot_df, 
        x='age', 
        hue='Group', 
        multiple='dodge',  # This puts bars side-by-side
        shrink=0.8,        # Adds a small gap between groups of bars for readability
        common_norm=False, # Set to False to see actual counts rather than proportions
        palette={'With PD': 'blue', 'Without PD': 'green', 'Total': 'gray'},
        alpha=0.4,
        bins=bin_edges
    )

    # Set the ticks to the midpoints and apply the labels
    plt.xticks(bin_midpoints, bin_labels, size=14)

    # Increase Legend Font Size
    # prop={'size': 14} handles the group names, title_fontsize handles 'Group'
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax.get_legend().get_title(), fontsize='16')

    # plt.title('Age Distribution: With PD vs Without PD vs Total')
    plt.xlabel('Age', size=16)
    plt.ylabel('Frequency (Count)', size=16)
    plt.grid(axis='y', alpha=0.3)

    # Save or display the plot
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'age_distribution_histogram.png'), dpi=300, bbox_inches='tight')
    return

def plot_sex_distribution(data_df):
    # print(df["pd"].unique())  # ['no' 'yes']
    df = data_df.drop_duplicates(subset=['Participant_ID'])

    # 1. Prepare the subgroups
    df_with_pd = df[df['pd'] == 'yes'].copy()
    df_with_pd['Group'] = 'With PD'

    df_without_pd = df[df['pd'] == 'no'].copy()
    df_without_pd['Group'] = 'Without PD'

    # 2. Prepare the "Total" group by duplicating the whole dataset
    df_total = df.copy()
    df_total['Group'] = 'Total'

    # 3. Concatenate the groups into one DataFrame for plotting
    plot_df = pd.concat([df_with_pd, df_without_pd, df_total]).reset_index(drop=True)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=plot_df, 
        x='gender', 
        hue='Group', 
        multiple='dodge',  # This puts bars side-by-side
        shrink=0.8,        # Adds a small gap between groups of bars for readability
        common_norm=False, # Set to False to see actual counts rather than proportions
        palette={'With PD': 'blue', 'Without PD': 'green', 'Total': 'gray'},
        alpha=0.4,
        bins=len(plot_df['gender'].unique())
    )

    # Set the ticks to the midpoints and apply the labels
    # plt.xticks(bin_midpoints, bin_labels, size=14)

    # Increase Legend Font Size
    # prop={'size': 14} handles the group names, title_fontsize handles 'Group'
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax.get_legend().get_title(), fontsize='16')

    # plt.title('Age Distribution: With PD vs Without PD vs Total')
    plt.xlabel('Sex', size=16)
    plt.ylabel('Frequency (Count)', size=16)
    plt.grid(axis='y', alpha=0.3)

    # Save or display the plot
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'sex_distribution_histogram.png'), dpi=300, bbox_inches='tight')

    # Create a latex table
    # Create a categorical column for the Sex values in your plot_df
    plot_df['Sex'] = plot_df['gender']

    # 2. Create the table using crosstab
    # This counts how many participants fall into each Sex per Group
    sex_table = pd.crosstab(plot_df['Sex'], plot_df['Group'])
    # 3. Optional: Reorder columns to match your plot's visual order
    column_order = ['With PD', 'Without PD', 'Total']
    sex_table = sex_table[column_order]
    sex_table.to_csv(os.path.join(CSV_SAVE_DIR, 'sex_distribution_table.csv'))

    latex_code = sex_table.to_latex(
        index=True, 
        label="tab:sex_dist",
        column_format='lccc',  # l=left (index), ccc=centered columns
        bold_rows=True,
        multicolumn_format='c'
    )

    with open(os.path.join(LATEX_SAVE_DIR, 'sex_distribution_table.tex'), 'w') as f:
        f.write(latex_code)

    return

def plot_race_distribution(data_df):
    # print(df["pd"].unique())  # ['no' 'yes']
    df = data_df.drop_duplicates(subset=['Participant_ID'])

    # 1. Prepare the subgroups
    df_with_pd = df[df['pd'] == 'yes'].copy()
    df_with_pd['Group'] = 'With PD'

    df_without_pd = df[df['pd'] == 'no'].copy()
    df_without_pd['Group'] = 'Without PD'

    # 2. Prepare the "Total" group by duplicating the whole dataset
    df_total = df.copy()
    df_total['Group'] = 'Total'

    # 3. Concatenate the groups into one DataFrame for plotting
    plot_df = pd.concat([df_with_pd, df_without_pd, df_total]).reset_index(drop=True)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=plot_df, 
        x='race', 
        hue='Group', 
        multiple='dodge',  # This puts bars side-by-side
        shrink=0.8,        # Adds a small gap between groups of bars for readability
        common_norm=False, # Set to False to see actual counts rather than proportions
        palette={'With PD': 'blue', 'Without PD': 'green', 'Total': 'gray'},
        alpha=0.4,
        bins=len(plot_df['race'].unique())
    )

    # Set the ticks to the midpoints and apply the labels
    # plt.xticks(bin_midpoints, bin_labels, size=14)

    # Increase Legend Font Size
    # prop={'size': 14} handles the group names, title_fontsize handles 'Group'
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax.get_legend().get_title(), fontsize='16')

    # plt.title('Age Distribution: With PD vs Without PD vs Total')
    plt.xlabel('Race', size=16)
    plt.ylabel('Frequency (Count)', size=16)
    plt.grid(axis='y', alpha=0.3)

    # Save or display the plot
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'race_distribution_histogram.png'), dpi=300, bbox_inches='tight')

    # Create a latex table
    # Create a categorical column for the Sex values in your plot_df
    plot_df['Race'] = plot_df['race']

    # 2. Create the table using crosstab
    # This counts how many participants fall into each Race per Group
    race_table = pd.crosstab(plot_df['Race'], plot_df['Group'])
    # 3. Optional: Reorder columns to match your plot's visual order
    column_order = ['With PD', 'Without PD', 'Total']
    race_table = race_table[column_order]
    race_table.to_csv(os.path.join(CSV_SAVE_DIR, 'race_distribution_table.csv'))
    latex_code = race_table.to_latex(
        index=True, 
        label="tab:race_dist",
        column_format='lccc',  # l=left (index), ccc=centered columns
        bold_rows=True,
        multicolumn_format='c'
    )

    with open(os.path.join(LATEX_SAVE_DIR, 'race_distribution_table.tex'), 'w') as f:
        f.write(latex_code)

    return

def plot_pd_stage_distribution(data_df):
    # Only keep participants with PD
    df = data_df.drop_duplicates(subset=['Participant_ID'])
    df = df[df['pd'] == 'yes']
    plot_df = df

    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(
        data=plot_df, 
        x='PD_Stage',
        discrete=True, 
        shrink=0.8,
        common_norm=False, # Set to False to see actual counts rather than proportions
        alpha=0.4
    )
    
    # Set the ticks to the midpoints and apply the labels
    plt.xticks([1, 2, 3], ['Stage 1', 'Stage 2', 'Stage 3'], size=14)

    plt.xlabel('PD Stage', size=16)
    plt.ylabel('Frequency (Count)', size=16)
    plt.grid(axis='y', alpha=0.3)

    # Save or display the plot
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'pd_stage_distribution_histogram.png'), dpi=300, bbox_inches='tight')

    # # Create a latex table
    # # Create a categorical column for the Sex values in your plot_df
    # plot_df['Sex'] = plot_df['gender']

    # # 2. Create the table using crosstab
    # # This counts how many participants fall into each Sex per Group
    # sex_table = pd.crosstab(plot_df['Sex'], plot_df['Group'])
    # # 3. Optional: Reorder columns to match your plot's visual order
    # column_order = ['With PD', 'Without PD', 'Total']
    # sex_table = sex_table[column_order]
    # sex_table.to_csv(os.path.join(CSV_SAVE_DIR, 'sex_distribution_table.csv'))

    # latex_code = sex_table.to_latex(
    #     index=True, 
    #     label="tab:sex_dist",
    #     column_format='lccc',  # l=left (index), ccc=centered columns
    #     bold_rows=True,
    #     multicolumn_format='c'
    # )

    # with open(os.path.join(LATEX_SAVE_DIR, 'sex_distribution_table.tex'), 'w') as f:
    #     f.write(latex_code)

    return

def plot_task_distribution(data_df):
    # print(df["pd"].unique())  # ['no' 'yes']
    # df = data_df.drop_duplicates(subset=['Participant_ID'])
    df = data_df.copy()

    # Standardize Task names
    for task in task_name_mapping.keys():
        df.loc[df['Task'].isin(task_name_mapping[task]), 'Task'] = task

    # 1. Prepare the subgroups
    df_with_pd = df[df['pd'] == 'yes'].copy()
    df_with_pd['Group'] = 'With PD'

    df_without_pd = df[df['pd'] == 'no'].copy()
    df_without_pd['Group'] = 'Without PD'

    # 2. Prepare the "Total" group by duplicating the whole dataset
    df_total = df.copy()
    df_total['Group'] = 'Total'

    # 3. Concatenate the groups into one DataFrame for plotting
    plot_df = pd.concat([df_with_pd, df_without_pd, df_total]).reset_index(drop=True)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=plot_df, 
        x='Task',
        discrete=True, 
        hue='Group', 
        multiple='dodge',  # This puts bars side-by-side
        shrink=0.8,        # Adds a small gap between groups of bars for readability
        common_norm=False, # Set to False to see actual counts rather than proportions
        palette={'With PD': 'blue', 'Without PD': 'green', 'Total': 'gray'},
        alpha=0.4,
    )

    # Set the ticks to the midpoints and apply the labels
    # plt.xticks(bin_midpoints, bin_labels, size=14)

    # Increase Legend Font Size
    # prop={'size': 14} handles the group names, title_fontsize handles 'Group'
    plt.setp(ax.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax.get_legend().get_title(), fontsize='16')

    # plt.title('Age Distribution: With PD vs Without PD vs Total')
    plt.xlabel('Task', size=16)
    plt.ylabel('Number of Videos', size=16)
    plt.grid(axis='y', alpha=0.3)

    # Save or display the plot
    plt.savefig(os.path.join(PLOT_SAVE_DIR, 'task_distribution_histogram.png'), dpi=300, bbox_inches='tight')

    # Create a latex table
    
    # 1. Create the table using crosstab
    # This counts how many participants fall into each Task per Group
    task_table = pd.crosstab(plot_df['Task'], plot_df['Group'])
    task_table.loc['Grand Total'] = task_table.sum()
    
    # 2. Optional: Reorder columns to match your plot's visual order
    column_order = ['With PD', 'Without PD', 'Total']
    task_table = task_table[column_order]
    task_table.to_csv(os.path.join(CSV_SAVE_DIR, 'task_distribution_table.csv'))
    latex_code = task_table.to_latex(
        index=True, 
        label="tab:task_dist",
        column_format='lccc',  # l=left (index), ccc=centered columns
        bold_rows=False,
        multicolumn_format='c'
    )

    with open(os.path.join(LATEX_SAVE_DIR, 'task_distribution_table.tex'), 'w') as f:
        f.write(latex_code)

    return

if __name__ == "__main__":
    data_df = clean_data()
    # Columns: 
    # 'Filename', 'Protocol', 'Participant_ID', 'Task', 'Duration', 'FPS',
    # 'Frame_Height', 'Frame_Width', 'gender', 'age', 'race', 'ethnicity',
    # 'pd', 'dob', 'time_mdsupdrs'

    # Standardize PD labels as screening outcomes
    # no = no
    # yes = yes
    # Unlikely = no
    # Possible = yes
    # Probable = yes
    data_df.loc[data_df['pd']=='Unlikely', 'pd'] = 'no'
    data_df.loc[data_df['pd']=='Possible', 'pd'] = 'yes'
    data_df.loc[data_df['pd']=='Probable', 'pd'] = 'yes'
    pd_labels = data_df["pd"].unique() # ['no' 'yes' 'Unlikely' 'Possible' 'Probable']
    print(f"PD labels found in dataset (after filtering None values): {pd_labels}")
    newline()

    plot_age_distribution(data_df)
    plot_sex_distribution(data_df)
    plot_race_distribution(data_df)
    plot_pd_stage_distribution(data_df)
    plot_task_distribution(data_df)

    # Basic statistics
    print("Data frequency across protocols:")
    total = 0
    for x in data_df["Protocol"].unique():
        num_videos = data_df[data_df["Protocol"]==x].shape[0]
        total += num_videos
        print(f"Number of videos for protocol {x}: {num_videos}")
    print(f"Total videos counted: {total}")
    newline()

    # # Task-wise video count
    # print("Data frequency across tasks:")
    # total_data = 0
    # total_subjects = 0
    # for x in task_name_mapping.keys():
    #     task_subset = data_df[data_df["Task"].isin(task_name_mapping[x])]
    #     num_videos = task_subset.shape[0]
    #     num_subjects = task_subset["Participant_ID"].nunique()
    #     total_data += num_videos
    #     total_subjects += num_subjects
    #     print(f"Number of videos for task {x}: {num_videos}")
    #     print(f"Number of unique subjects for task {x}: {num_subjects}")
    #     print("##")
    # print(f"Total videos counted: {total_data}")
    # print(f"Total unique subjects counted across all tasks: {total_subjects}")
    # newline()

    # print("Data frequency across PD/Non-PD groups:")
    # total = 0
    # for x in data_df["pd"].unique():
    #     num_videos = data_df[data_df["pd"]==x].shape[0]
    #     total += num_videos
    #     print(f"Number of videos for PD status {x}: {num_videos}")
    # print(f"Total videos counted: {total}")
    # newline()
    
    # print(data_df.columns)

    # End of program
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    os.rename("/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/general_statistics_temp.log", "/localdisk1/PARK/park_video_benchmarking/results/R1_Dataset/general_statistics.log")