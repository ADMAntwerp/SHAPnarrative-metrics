import dill
import numpy as np
from evaluate import load
import argparse 
import transformers
from shapnarrative_metrics.metrics.perplexity import compute_perplexity
from shapnarrative_metrics.metrics.bleurt import compute_BLEURT


transformers.logging.set_verbosity_info()

"""This script adds the metrics that require some more computation and are best run on the cloud.
   Here, perplexities and BLEURT will be added to the range of experiments for which the local metrics have already been computed.
   HF token should be passed in terminal 
"""

N_range=2
experiment_dir="standard_experiments"
# experiment_dir="manipulated_experiments"

METRICS_PATHS=[f"results/{experiment_dir}/experiment_{i}/metrics.pkl" for i in range(1,N_range)]
SAVE_PATHS=[f"results/{experiment_dir}/experiment_{i}/metrics_cloud.pkl" for i in range(1,N_range)]

BLEURT_MODEL="BLEURT-20"
MODELS=["meta-llama/Meta-Llama-3-8B","mistralai/Mistral-7B-v0.3"]
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Generate a series of narratives and save them")
    parser.add_argument("--HF_token", default="", type=str, help=f"huggingface token for lama-3 8b base or Mistral 7b")
    parser.add_argument("--METRICS_PATHS", '--experiment_paths_list' , nargs='+', default=METRICS_PATHS, help=f"Path to dir to read metrics from to which they will also be saved")
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, help=f"Path to dir to where metrics after cloud will be saved")


    args=parser.parse_args()

    for metrics_path in args.METRICS_PATHS: 

        with open(metrics_path, "rb") as f:
            metrics=dill.load(f)


        #add both faithfulness and embedding distances
        metrics=compute_perplexity(metrics, MODELS,args.HF_token)

        bleurt = load("bleurt", BLEURT_MODEL)
        metrics=compute_BLEURT(metrics)

        with open(metrics_path, "wb") as f:
            dill.dump(metrics, f)
    
