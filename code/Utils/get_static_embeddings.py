import os
import pickle
import torch
import pandas as pd


model_embedding_paths = {
    "ViViT": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/ViViT/ViViT_Features_All_PARK_Videos.pkl",
    "VideoMAE": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/VideoMAE/VideoMAE_Features_All_PARK_Videos.pkl",
    "TimeSformer": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/TimeSformer/TimeSformer_Features_All_PARK_Videos.pkl",
    "VideoPrism": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/VideoPrism/VideoPrism_Features_All_PARK_Videos.pkl",
    "VideoMAEv2": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/VideoMAEv2/VideoMAEv2_Features_All_PARK_Videos.pkl",
    "VJEPA2": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/VJEPA2/VJEPA2_Features_All_PARK_Videos.pkl",
    "VJEPA2_SSV2": "/localdisk1/PARK/park_video_benchmarking/data/single_view_embeddings/VJEPA2_SSV2/VJEPA2_Features_All_PARK_Videos.pkl"
}

model_embedding_paths_multiview = {
    4: {
        "ViViT": "/localdisk1/PARK/park_video_benchmarking/data/multi_view_embeddings/ViViT/ViViT_4views_2stride_Features_All_PARK_Videos.pkl",
        "VideoMAE": "/localdisk1/PARK/park_video_benchmarking/data/multi_view_embeddings/VideoMAE/VideoMAE_4views_2stride_Features_All_PARK_Videos.pkl",
        "TimeSformer": "/localdisk1/PARK/park_video_benchmarking/data/multi_view_embeddings/TimeSformer/TimeSformer_4views_2stride_Features_All_PARK_Videos.pkl",
        "VideoPrism": "/localdisk1/PARK/park_video_benchmarking/data/multi_view_embeddings/VideoPrism/VideoPrism_4views_2stride_Features_All_PARK_Videos.pkl",
        "VJEPA2": "/localdisk1/PARK/park_video_benchmarking/data/multi_view_embeddings/VJEPA2/VJEPA2_basic_4views_2stride_Features_All_PARK_Videos.pkl"
    }
}

def get_all_static_embeddings(model, num_views=1, view_index=0, pooling="mean"):
    """
    Get saved extracted embeddings.
    Supports single view and multi-view embeddings (n_views = 4)
    num_views = 1 --> single view # shape: [d_embedding]
    num_views = 4, view_index = {0, 1, 2, 3} --> one view from multi-view embeddings # shape: [d_embedding]
    num_views = 4, view_index = -1 --> return embeddings from all views # shape: [n_views, d_embedding]
    """
    df_features = None

    if num_views == 1:
        assert (model in model_embedding_paths.keys()), f"Unsupported single-view implementation for model {model}"
        embedding_saved_path = model_embedding_paths[model]
        with open(embedding_saved_path, 'rb') as f:
            loaded_data = pickle.load(f)
            df_features = pd.DataFrame.from_dict(loaded_data)
            
            if pooling == "mean":
                df_features = df_features.rename(columns={"filename":"file_name", "mean_pooled_embedding":"features"})
            elif pooling == "max":
                df_features = df_features.rename(columns={"filename":"file_name", "max_pooled_embedding":"features"})

            df_features = df_features[["file_name", "features"]]
    else:
        assert (model in model_embedding_paths_multiview[4].keys()), f"Unsupported multi-view implementation for model {model}"
        if num_views not in model_embedding_paths_multiview.keys():
            raise Exception(f"Multi-view extraction not implemented for num_views = {num_views}")
        
        embedding_saved_path = model_embedding_paths_multiview[num_views][model]
        
        if view_index == -1:
            '''
            Select all views as the feature
            '''
            with open(embedding_saved_path, 'rb') as f:
                loaded_data = pickle.load(f)
                df_features = pd.DataFrame.from_dict(loaded_data)

                if pooling == "mean":
                    if model == "VideoPrism":
                        df_features["features"] = df_features.apply(lambda x: torch.stack([torch.tensor(x.view_0_mean_pooled_embedding), torch.tensor(x.view_1_mean_pooled_embedding), torch.tensor(x.view_2_mean_pooled_embedding), torch.tensor(x.view_3_mean_pooled_embedding)]), axis=1)
                    else:
                        df_features["features"] = df_features.apply(lambda x: torch.stack([x.view_0_mean_pooled_embedding, x.view_1_mean_pooled_embedding, x.view_2_mean_pooled_embedding, x.view_3_mean_pooled_embedding]), axis=1)
                elif pooling == "max":
                    if model == "VideoPrism":
                        df_features["features"] = df_features.apply(lambda x: torch.stack([torch.tensor(x.view_0_max_pooled_embedding), torch.tensor(x.view_1_max_pooled_embedding), torch.tensor(x.view_2_max_pooled_embedding), torch.tensor(x.view_3_max_pooled_embedding)]), axis=1)
                    else:
                        df_features["features"] = df_features.apply(lambda x: torch.stack([x.view_0_max_pooled_embedding, x.view_1_max_pooled_embedding, x.view_2_max_pooled_embedding, x.view_3_max_pooled_embedding]), axis=1)
            
            df_features = df_features.rename(columns={"filename":"file_name"})
            df_features = df_features[["file_name", "features"]]
        else:
            with open(embedding_saved_path, 'rb') as f:
                loaded_data = pickle.load(f)
                df_features = pd.DataFrame.from_dict(loaded_data)
                
                if pooling == "mean":
                    df_features = df_features.rename(columns={"filename":"file_name", f"view_{view_index}_mean_pooled_embedding":"features"})
                elif pooling == "max":
                    df_features = df_features.rename(columns={"filename":"file_name", f"view_{view_index}_max_pooled_embedding":"features"})

                df_features = df_features[["file_name", "features"]]
    return df_features

if __name__ == "__main__":

    # for model_name in model_embedding_paths.keys():
    #     print(f"Testing {model_name} Embeddings")
    #     x = get_all_static_embeddings(model=model_name, num_views=1, pooling="mean")
    #     x_f = x.iloc[0]["features"]
    #     print(f"Single-view {model_name} Embedding Shape: {x_f.shape}")
    #     print("==============================")
        
    # print()

    for model_name in model_embedding_paths_multiview[4].keys():
        print(f"Testing multi-view {model_name} Embeddings")
        x = get_all_static_embeddings(model=model_name, num_views=4, view_index=-1, pooling="mean")
        x_f = x.iloc[0]["features"]
        print(f"Multi-view {model_name} Embedding Shape: {x_f.shape}")
        print("==============================")