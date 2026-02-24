import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

with open("../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

PLOT_PATH = f"/localdisk1/{project_dir}/{project_name}/results/R4_Model_Performance_Comparison/Plots"

# read csv 
def read_csv(file_path):
    df = pd.read_csv(file_path)
    # check if the name column has any values without the substring "sweep"
    if 'name' in df.columns:
        df = df[df['name'].str.contains("sweep", na=False)]
    print(f"Dataframe loaded from {file_path} with shape {df.shape}")
    return df

def cluster_creation(df_all_run):
    model_names = sorted(df_all_run['model'].unique().tolist())

    clusters = {
        "Visual Speech Kinematics": {
            "tasks": [
                "sustained_phonation_a",
                "sustained_phonation_e",
                "sustained_phonation_o",
                "pangram_utterance",
                "tongue_twister"
            ],
        },
        "Facial Expressivity": {
            "tasks": [
                "facial_expression_smile",
                "facial_expression_disgust",
                "facial_expression_surprise",
            ],
        },
        "Upper-Limb Motor Kinematics": {
            "tasks": [
                "finger_tapping",
                "open_fist",
                "flip_palm",
                "extend_arm",
                "nose_touch"
            ]
        },
        "Oculo-Cervical & Cognitive Control": {
            "tasks": [
                "eye_gaze",
                "head_pose",
                "reverse_count"
        ]
        }
    }
    for cluster_name in clusters:
        task_list = clusters[cluster_name]["tasks"]
        for model in model_names:
            clusters[cluster_name][model] = {}
            for task in task_list:
                clusters[cluster_name][model][task] = task
                # now get the test_auroc after sorting by best dev_auroc of the dataframe having same task and model name
                filtered_df = df_all_run[(df_all_run['model'] == model) & (df_all_run['task_name'] == task)]
                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values(by='dev_auroc', ascending=False)
                    test_auroc = filtered_df['test_auroc'].values[0]
                    clusters[cluster_name][model][task] = test_auroc
    return clusters

def plot_cluster_boxes_clean_x(
    cluster_results,
    task_label_map=None,
    output_path=None,
    gap=1.8,
    box_width=0.85,
    figsize=(16, 12),
    model_legend_ncol=7,
    task_legend_ncol=4,
):
    """
    cluster_results format:
      {
        "Cluster A": {
            "tasks": [task1, task2, ...],
            "Model1": {task1: auroc, task2: auroc, ...},
            "Model2": {...},
            ...
        },
        ...
      }

    Plot:
      - x-axis: cluster names only (clean)
      - within each cluster: one x-position per model (hidden on x-axis)
      - for each (cluster, model): translucent box spans min..max AUROC across tasks
      - task AUROCs plotted as markers at same x-position
      - model legend (colors) at top; task legend (markers) at bottom
    """

    # --- infer model list from the first cluster (excluding 'tasks') ---
    first_cluster = next(iter(cluster_results.values()))
    models = [k for k in first_cluster.keys() if k != "tasks"]

    # --- model colors (consistent) ---
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    # --- global task list -> marker mapping (consistent across clusters) ---
    all_tasks = []
    for c in cluster_results:
        all_tasks.extend(cluster_results[c]["tasks"])
    seen = set()
    all_tasks = [t for t in all_tasks if not (t in seen or seen.add(t))]
    
    if task_label_map is None:
        task_label_map = {t: t for t in all_tasks}

    # avoid line-only markers like '1','2','3' to prevent edgecolor warnings
    marker_cycle = ["o","s","^","D","P","X","*","v","<",">","h","H","d","p","8", "$\clubsuit$"]
    task_markers = {t: marker_cycle[i % len(marker_cycle)] for i, t in enumerate(all_tasks)}

    # --- x positions with gaps between clusters ---
    x_positions = {}
    cluster_centers = {}
    x = 0.0
    for cluster_name in cluster_results:
        start = x
        for m in models:
            x_positions[(cluster_name, m)] = x
            x += 1.0
        end = x - 1.0
        cluster_centers[cluster_name] = (start + end) / 2.0
        x += gap

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- draw boxes + task points ---
    global_min, global_max = 1.0, 0.0
    for cluster_name, content in cluster_results.items():
        tasks = content["tasks"]
        for m in models:
            if m not in content:
                continue

            scores = [content[m][t] for t in tasks if t in content[m]]
            if not scores:
                continue

            y_min, y_max = float(np.min(scores)), float(np.max(scores))
            global_min = min(global_min, y_min)
            global_max = max(global_max, y_max)

            xp = x_positions[(cluster_name, m)]

            # background min..max range box
            rect = patches.Rectangle(
                (xp - box_width / 2, y_min),
                box_width,
                max(0.001, y_max - y_min),
                facecolor=model_colors[m],
                edgecolor=model_colors[m],
                linewidth=1.0,
                alpha=0.18,
                zorder=1,
            )
            ax.add_patch(rect)
            
            # --- mean line inside the box (avg over tasks for this model within this cluster) ---
            y_mean = float(np.mean(scores))
            ax.hlines(
                y=y_mean,
                xmin=xp - box_width/2,
                xmax=xp + box_width/2,
                colors=model_colors[m],
                linewidth=2.0,
                zorder=2,          # above box, below points
                alpha=0.9
            )
            
            # task points at same x (distinct marker per task)
            for t in tasks:
                if t in content[m]:
                    # ax.scatter(
                    #     [xp], [content[m][t]],
                    #     marker=task_markers[t],
                    #     s=65,
                    #     color=model_colors[m],
                    #     edgecolors="black",
                    #     linewidths=0.4,
                    #     zorder=3,
                    # )

                    ax.scatter(
                        [xp], [content[m][t]],
                        marker=task_markers[t],
                        linewidths=1,
                        color=model_colors[m],
                        edgecolors="black",
                        alpha=0.5,
                        s=60,
                        zorder=3,
                    )
        # for each cluster, add a vertical separator line on the right edge (except for last cluster)
        if cluster_name != list(cluster_results.keys())[-1]:
            next_cluster = list(cluster_results.keys())[list(cluster_results.keys()).index(cluster_name) + 1]
            next_xp = x_positions[(next_cluster, models[0])]
            ax.axvline(x=next_xp - gap/2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # --- y formatting ---
    ax.set_ylabel("AUC", fontsize=18)
    ax.set_ylim(max(0.45, global_min - 0.03), min(1.0, global_max + 0.03))
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)

    # --- CLEAN x-axis: cluster labels only ---
    cluster_tick_positions = [cluster_centers[c] for c in cluster_results.keys()]
    cluster_tick_labels = list(cluster_results.keys())
    ax.set_xticks(cluster_tick_positions)
    ax.set_xticklabels(cluster_tick_labels, fontsize=16, rotation=10)
    ax.tick_params(axis="x", length=0)  # remove tick marks
    ax.tick_params(axis='y', labelsize=16)

    # ax.set_title("Clustered AUROC: per-model min–max box + per-task points", pad=28)

    model_map = {m: m for m in models}
    model_map['VJEPA2_SSV2'] = 'V-JEPA2-SSv2'
    model_map['VJEPA2'] = 'V-JEPA2'

    # print(model_map)
    # # --- legends ---
    # # Model legend (colors) at top
    # model_handles = [
    #     plt.Line2D(
    #         [0], [0],
    #         marker="s",
    #         linestyle="None",
    #         markersize=10,
    #         markerfacecolor=model_colors[m],
    #         markeredgecolor="black",
    #         label=model_map[m]
    #     )
    #     for m in models
    # ]

    model_handles = [
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markersize=10,
            markerfacecolor=model_colors[m],
            markeredgecolor=model_colors[m],
            alpha=1,            # <-- MATCH box transparency
            label=model_map.get(m, m)
        )
        for m in models
    ]

    model_legend = ax.legend(
        handles=model_handles,
        # title="Model (color)",
        ncol=model_legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),   
        frameon=False,
        fontsize=14,
    )
    ax.add_artist(model_legend)

    # Convert data-x to axes-fraction for bbox_to_anchor
    xmin, xmax = ax.get_xlim()
    def xdata_to_axfrac(xdata):
        return (xdata - xmin) / (xmax - xmin)

    cluster_task_legends = []
    y_anchor = -0.18   # vertical position under the x-axis (tune if needed)

    start_x = 0.18  # Adjust based on your first box

    # The fixed distance between the centers of each box
    spacing = {
        0: 0.25,
        1: 0.24,
        2: 0.26,
        3: 0.25
    }
    
    cluster_index = 0
    space = 0

    for cluster_name, content in cluster_results.items():
        tasks = content["tasks"]

        x_pos = start_x + space
        space += spacing[cluster_index]
        
        handles = [
            plt.Line2D(
                [0], [0],
                marker=task_markers[t],
                linestyle="None",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="black",
                label=task_label_map.get(t, t)
            )
            for t in tasks
        ]

        # leg = ax.legend(
        #     handles=handles,
        #     ncol=1 if len(handles) <= 3 else 2,   # auto 1/2 columns per cluster
        #     loc="upper center",
        #     bbox_to_anchor=(xdata_to_axfrac(cluster_centers[cluster_name]), y_anchor),
        #     frameon=True,
        #     fontsize=14,
        #     handletextpad=0.6,
        #     columnspacing=1.0,
        #     borderaxespad=0.0,
        # )

        leg = ax.legend(
            handles=handles,
            ncol=1 if len(handles) <= 3 else 2,   # auto 1/2 columns per cluster
            loc="upper center",
            bbox_to_anchor=(x_pos, y_anchor),
            frameon=True,
            fontsize=14,
            handletextpad=0.6,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        
        # Modify the frame's line style
        frame = leg.get_frame()
        frame.set_linestyle('--')  # Options: '--', 'dashed', 'dotted', etc.
        frame.set_edgecolor('black') # Optional: ensure the color is visible

        ax.add_artist(leg)
        cluster_task_legends.append(leg)
        cluster_index += 1

    # Increase bottom margin to make room for per-cluster legends
    fig.subplots_adjust(top=0.82, bottom=0.32)

    # Save without cropping (include all legends)
    fig.canvas.draw()

    if output_path:
        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            bbox_extra_artists=[model_legend] + cluster_task_legends,
            pad_inches=0.35
        )

    plt.show()

def main(file_path, tag="single_view"):
    df_all_run = read_csv(file_path)
    clusters = cluster_creation(df_all_run)

    # Human-readable labels (same wording as legend style)
    task_label_map = {
        "sustained_phonation_a": "Phonation /a/",
        "sustained_phonation_e": "Phonation /e/",
        "sustained_phonation_o": "Phonation /o/",
        "pangram_utterance": "Pangram",
        "tongue_twister": "Tongue twister",
        "facial_expression_smile": "Smile",
        "facial_expression_disgust": "Disgust",
        "facial_expression_surprise": "Surprise",
        "resting_face": "Resting face",
        "finger_tapping": "Finger tapping",
        "open_fist": "Open fist",
        "flip_palm": "Flip palm",
        "extend_arm": "Extend arm",
        "nose_touch": "Nose touch",
        "eye_gaze": "Eye gaze",
        "head_pose": "Head pose",
        "reverse_count": "Reverse count",
    }
    output_path = os.path.join(PLOT_PATH, f"clustered_auroc_{tag}.png")
    plot_cluster_boxes_clean_x(clusters, task_label_map=task_label_map, output_path=output_path)
    return

if __name__ == "__main__":
    file_path = f"/localdisk1/{project_dir}/{project_name}/results/R2_Task_Screening_Performance/wandb_results/wandb_runs_summary_all_runs.csv"
    main(file_path)



