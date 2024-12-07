import dill
from shapnarrative_metrics.metrics.faithfulness import compute_faithfulness
from shapnarrative_metrics.metrics.embedding_distance import compute_embedding_distance
import argparse


"""
In this script we start from the generated experiments from run_experiment.py and add local metrics -- faithfulness and cosine       similarity distance (=no VM or cloud required). 
"""

N_range=5
# experiment_dir="longshort_experiments"
experiment_dir="manipulated_experiments"
# experiment_dir="manipulated_experiments"

EXPERIMENT_PATHS=[f"results/{experiment_dir}/experiment_{i}/experiment.pkl" for i in range(1,N_range)]
SAVE_PATHS=[f"results/{experiment_dir}/experiment_{i}/metrics.pkl" for i in range(1,N_range)]
 
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Generate a series of narratives and save them")
    parser.add_argument("--EXPERIMENT_PATHS", '--experiment_paths_list' , nargs='+', default=EXPERIMENT_PATHS, type=list, help=f"Path to dir to read experiments from")
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, type=list, help=f"Path to dir where the metrics will be saved")

    args=parser.parse_args()

    for experiment_path, save_path in zip(args.EXPERIMENT_PATHS, args.SAVE_PATHS):

        with open(experiment_path, "rb") as f:
            experiments=dill.load(f)


        #add both faithfulness and embedding distances
        metrics=compute_faithfulness(experiments)
        metrics=compute_embedding_distance(metrics)

        with open(save_path, "wb") as f:
            dill.dump(metrics, f)
    