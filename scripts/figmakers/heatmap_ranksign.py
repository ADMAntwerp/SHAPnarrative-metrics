 
import argparse
from statistics import mean, median
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Type
import dill

experiment_dir="longshort_experiments"
METRICS_PATH=f"results/{experiment_dir}/experiment_1/metrics.pkl" 
PROMPT_TYPE="long"

def dfs_to_heatmap(df_rank: pd.DataFrame, df_sign: pd.DataFrame, df_value: pd.DataFrame, show_window=6):

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    cmap=sns.color_palette("rocket_r", as_cmap=True)
    cmap2 = plt.cm.coolwarm

    sns.heatmap(
                df_rank,
                cmap=cmap2,
                linewidths=0.5,
                annot=True,
                mask=np.isinf(df_rank),
                ax=axes[0],
                vmin=-3,
                vmax=3,
                xticklabels=False)  # Set the first heatmap to the first subplot

    sns.heatmap(df_sign, 
                cmap=cmap, 
                linewidths=0.5, 
                annot=True, 
                mask=np.isinf(df_sign),
                ax=axes[1],
                vmin=0,
                vmax=5,
                xticklabels=False)  # Set the second heatmap to the second subplot
    sns.heatmap(df_value, 
            cmap=cmap2, 
            linewidths=0.5, 
            annot=True, 
            mask=np.isinf(df_value),
            ax=axes[2],
            vmin=-3,
            vmax=3,
            xticklabels=False)  # Set the second heatmap to the second subplot
    # Add titles
    axes[0].set_title('Rank difference', fontsize=16)
    axes[0].set_xlabel('Extracted feature rank', fontsize=12)
    axes[0].set_ylabel('Narratives', fontsize=12)

    axes[1].set_title('Sign difference (abs)', fontsize=16)
    axes[1].set_xlabel('Extracted feature rank', fontsize=12)
    axes[1].set_ylabel('Narratives', fontsize=12)

    axes[2].set_title('Value difference (abs)', fontsize=16)
    axes[2].set_xlabel('Extracted feature rank', fontsize=12)
    axes[2].set_ylabel('Narratives', fontsize=12)

    axes[0].set_xlim([0, show_window])
    axes[1].set_xlim([0, show_window])
    axes[2].set_xlim([0, show_window])

    plt.tight_layout()  # Adjust layout to prevent overlapping
    
    return fig

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--METRICS_PATH", default=METRICS_PATH, type=str, help=f"Path to experiment pickled list with metrics")
    args=parser.parse_args()

    with open(args.METRICS_PATH, "rb") as f:
        metrics=dill.load(f)

    for metric in metrics:
        
        if metric.prompt_type==PROMPT_TYPE:

            for extraction_model in metric.extractions_dict.keys():
                fig_ranksign=dfs_to_heatmap(metric.rank_diff[extraction_model],metric.sign_diff[extraction_model], metric.value_diff[extraction_model])

            plt.savefig(f"results/figures/heatmaps/heatmap_{metric.dataset}_{metric.generation_model}.png", dpi=300, bbox_inches='tight')

