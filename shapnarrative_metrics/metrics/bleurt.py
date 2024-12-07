import dill
import numpy as np
from evaluate import load
import argparse 
from shapnarrative_metrics.experiment_management.experiment_dataclass import ExperimentMetrics
from dataclasses import dataclass, asdict, field
from evaluate import EvaluationModule
from typing import Type, Tuple
import transformers

transformers.logging.set_verbosity_info()


def filter_experiments(id: Tuple, experiments: list) -> Type[ExperimentMetrics]:

    
    filtered_experiments=[experiment for experiment in experiments if experiment.id==id]

    if len(filtered_experiments)!=1:
        print(f"Something wrong length of filtered experiments is {len(filtered_experiments)}")
    
    return filtered_experiments[0]


def HF_compute_metric(hf_metric: Type[EvaluationModule], llm_metric: Type[ExperimentMetrics], hu_metric: Type[ExperimentMetrics]) -> list[float]:

    """
    Given a huggingface metric class (in this case BLEURT) and two LLM and HU experiments (ExperimentMetrics instances) computes their BLEURT
    Args:
        hf_metric: HuggingFace initialized metric object that can be called to compute LLM-related metrics
        llm_metric: Metric object from our own repo that contains the results of an experiment that generated LLM narratives for a given dataset
        hu_metric: The corresponding human metric or experiment object
    
    Returns:
        list of floats that contain the corresponding bleurts between the narratives
    """
    
    
    #make sure that all LLM and all HU narratives in the list are in the same order
    assert llm_metric.idx_list==hu_metric.idx_list, "ORDER OF HU AND LLM NARRATIVES NOT THE SAME IN BLEURT -- CHECK"

    #next, compute and return the list of HF metrics between the two lists of narratives (bleurt)
    results=hf_metric.compute(predictions=llm_metric.narrative_list, references=hu_metric.narrative_list)

    return results["scores"]


def compute_BLEURT(bleurt, metrics):


    """Given a list of metrics, for every experiment computes the BLEURT distances between every narrative and the corresponding human narrative.
    Args:
        bleurt: a HF initiated bleurt object 
        metrics: list of ExperimentMetrics objects
    Returns:
        metrics: list of ExperimentMetrics with BLEURT lists added
    """

    for count, metric in enumerate(metrics):

        #for every metric load the corresponding human experiment
        hu_metric=filter_experiments((metric.dataset, metric.tar_model_name, "human", "long"), metrics ) 

        bleurt_list=HF_compute_metric(bleurt, hu_metric, metric)

        print(f"added bleurt to: {count}/{len(metrics)}")

        metric.bleurt=bleurt_list
    
    return metrics