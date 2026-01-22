import pickle
import pandas as pd

# embedding_saved_path = "/localdisk4/Ekram/VideoEncoderProject/datasets/videomae_v2/VideoMAEv2/VideoMAEv2_Features_All_PARK_Videos.pkl"
# embedding_saved_path = "/localdisk4/Ekram/VideoEncoderProject/datasets/vivit_based_features/raw_features/ViViT_Features_All_PARK_Videos.pkl"
# embedding_saved_path = "/localdisk4/Ekram/VideoEncoderProject/datasets/videoprism_large/VideoPrism/VideoPrism_Features_All_PARK_Videos.pkl"
# embedding_saved_path = "/localdisk4/Ekram/VideoEncoderProject/datasets/vjepa2_based_features/raw_based_features/VJEPA2_Features_All_PARK_Videos.pkl"
# embedding_saved_path = "/localdisk4/Ekram/VideoEncoderProject/datasets/vjepa2_vitg384/VJEPA2/VJEPA2_Features_All_PARK_Videos.pkl"
embedding_saved_path = "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/ViViT/ViViT_Features_All_PARK_Videos.pkl"

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