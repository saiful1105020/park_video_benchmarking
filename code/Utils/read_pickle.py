import pickle
import pandas as pd
import os

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

with open("../protocol_name.txt", "r") as f:
    protocol_name = f.read().strip()

embedding_saved_path = f"/localdisk1/{project_dir}/{project_name}/data/single_view_embeddings/ViViT/ViViT_Features_All_{project_dir}_Videos.pkl"

pooling = "mean"
with open(embedding_saved_path, 'rb') as f:
    loaded_data = pickle.load(f)
    df_features = pd.DataFrame.from_dict(loaded_data)
    
    if pooling == "mean":
        df_features = df_features.rename(columns={"filename":"file_name", "mean_pooled_embedding":"features"})
    elif pooling == "max":
        df_features = df_features.rename(columns={"filename":"file_name", "max_pooled_embedding":"features"})

    df_features = df_features[["file_name", "features"]]

print(df_features.head())
print(df_features['features'].iloc[0].shape)