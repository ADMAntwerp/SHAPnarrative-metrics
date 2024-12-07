import pickle
import dill
import random
import pandas as pd 
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load
import argparse 
from dataclasses import dataclass, asdict, field
from evaluate import EvaluationModule
from typing import Type, Tuple


experiment_name="experiment_1"
experiment_dir="standard_experiments"
METRICS_PATH=f"results/{experiment_dir}/{experiment_name}/metrics.pkl"
SAVE_DIR=f"results/figures/metrics_validation"
BLEURT_MODEL="BLEURT-20"


def filter_experiments(id, experiments: list):

    
    filtered_experiments=[experiment for experiment in experiments if experiment.id==id]

    if len(filtered_experiments)!=1:
        print(f"Something wrong length of filtered experiments is {len(filtered_experiments)}")
    return filtered_experiments[0]



if __name__=="__main__":

    
    parser = argparse.ArgumentParser(description="Compute perplexity for the extracted assumptions and their manipulated versions")
    parser.add_argument("--BLEURT_MODEL", default=BLEURT_MODEL, type=str, help=f"type of bleurt model")
    parser.add_argument("--METRICS_PATH", default=METRICS_PATH, type=str, help=f"path to load a metrics classes on top of which perplexity will be added")
    parser.add_argument("--SAVE_PATH", default=SAVE_DIR, type=str, help=f"path to save the metrics classes where perplexity has been added")

    args=parser.parse_args()


    with open(args.METRICS_PATH, "rb") as f:
        metrics=dill.load(f)


    #LOAD BLEURT 
    bleurt = load("bleurt", args.BLEURT_MODEL)
        
    def embedding_fig(X,Y,ds_classes):

        scatter_data = {
        'Index': [],
        'Distance': [],
        'Class':[]
        }
        matching_data={
        'Index': [],
        'Distance': [],
        'Class': []
        }
        counter=0
        for i, x in enumerate(X):
            for j,y in enumerate(Y):
            
                scatter_data['Index'].append(i)
                scatter_data['Distance'].append(bleurt.compute(predictions=[x],references=[y])["scores"][0])
                scatter_data['Class'].append(ds_classes[j])
                counter+=1
                if counter%100==0:
                    print(counter)

        for i, (x, y) in enumerate(zip(X, Y)):
            matching_data['Index'].append(i)
            matching_data['Distance'].append(bleurt.compute(predictions=[x],references=[y])["scores"][0])
            matching_data['Class'].append(ds_classes[i])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Index', y='Distance', data=scatter_data,hue="Class")
        sns.scatterplot(x='Index', y='Distance', data=matching_data,  color="magenta",marker="s",s=100, label="Matching narrative")

        ax.set_xlabel('Human narratives')
        ax.set_ylabel('Distance to LLM narr.')
        # ax.set_title('Scatter Plot of f(X, Y) for Each Index in X and Y in Y')
        # ax.legend(title='')
        ax.legend(loc="upper right")

        return fig, scatter_data, matching_data
    
    
    gen_model="gpt-4o"
    prompt_type="long"
    tar_model="RF"


    bleurt_values=[]
    embedding_model='voyage-large-2-instruct'
    dataset_order=["fifa","student","credit"]
    ds_classes=["fifa"]*20+["student"]*20+["credit"]*20

    llm_narratives=[]
    hu_narratives=[]

    for dataset in dataset_order:

        id=(dataset,tar_model,gen_model, prompt_type)
        id_hu=(dataset,tar_model,"human", prompt_type)

        metric=filter_experiments(id, metrics)
        hu_benchmark=filter_experiments(id_hu, metrics ) 
    
        
        llm_narratives+=metric.narrative_list
        hu_narratives+=hu_benchmark.narrative_list

    fig, scatter_data, matching_data=embedding_fig(hu_narratives, llm_narratives,ds_classes)


    scatter_df = pd.DataFrame(scatter_data)
    scatter_df.to_csv(f'{args.SAVE_DIR}/bleurt_scatter_data.csv', index=False)

    matching_df = pd.DataFrame(matching_data)
    matching_df.to_csv(f'{args.SAVE_DIR}/bleurt_matching_data.csv', index=False)

