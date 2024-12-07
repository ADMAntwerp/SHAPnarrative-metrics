import numpy as np
import pandas as pd
import pickle
import dill
from dataclasses import dataclass, asdict
from shapnarrative_metrics.llm_tools.extraction import ExtractionModel
from shapnarrative_metrics.experiment_management.experiment_dataclass import NarrativeExperiment, ExperimentMetrics
from typing import Union, Type, Tuple
import argparse
from scipy.stats import kendalltau
from scipy.spatial import distance

def filter_experiments(id: Tuple, experiments: list) -> Type[ExperimentMetrics]:

    
    filtered_experiments=[experiment for experiment in experiments if experiment.id==id]

    if len(filtered_experiments)!=1:
        print(f"Something wrong length of filtered experiments is {len(filtered_experiments)}")
    
    return filtered_experiments[0]


def compute_embedding_distance(experiments: list[Type[NarrativeExperiment]])->list[Type[ExperimentMetrics]]:
    
    """Takes a list of experiments and computes for every experiment the cosine similarity distance to the human narratives

    Args:
        experiments (list): A list of NarrativeExperiment objects containing the narratives and their parameters.
                            The objects in the list can also be ExperimentMetrics instances, but then embeddings distance will be overwritten.
    
    Returns:
        metrics (list): A list of ExperimentMetrics objects where embedding_distance as a cosine similarity distance to the human narratives has been added.
    """

    metrics=[]

    #ensure that the human narratives have been indeed added to the experiment list
    assert "human" in [exp.generation_model for exp in experiments], "HUMAN NARRATIVES NOT FOUND IN EXPERIMENT"


    #run over every experiment in the input, filter out the corresponding human experiment, and then compute the cosine similarity distance
    for experiment in experiments:

        embedding_distance_dict={}
        hu_benchmark_exp=filter_experiments((experiment.dataset, experiment.tar_model_name, "human", "long"), experiments ) 

        embedding_distance_list=[]
        for embedding_model, embedding_list in experiment.embeddings_dict.items():
            hu_embeddings=hu_benchmark_exp.embeddings_dict[embedding_model]
            embedding_distance_list=[ distance.cosine(emb1, emb2) for emb1, emb2 in zip(hu_embeddings, embedding_list) ]

            #we immediately take the mean over all the embedding distances per experiment so we really get 1 number per experiment
            embedding_distance_dict[embedding_model]=np.mean(embedding_distance_list)

        metric=ExperimentMetrics(
                                    **asdict(experiment), 
        )
        metric.embedding_distance=embedding_distance_dict
        metrics.append(metric)
    
    return metrics