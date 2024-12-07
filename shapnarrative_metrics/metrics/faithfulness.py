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


def average_zero(df):

    "compute the average occurrence of zeros in a dataframe among all numeric values"
    
    #take all values of the dataframe together
    values = df.values.flatten()

    #keep only array of numeric values (so completely ignore nans or np.infs)
    numeric_values = values[np.isfinite(values)]

    #count total zeroes and total numerics values in the df
    num_zeros = np.sum(numeric_values == 0)
    total_numeric_values = len(numeric_values)

    #compute the accuracy for those objects
    average_occurrence_of_zero = num_zeros / total_numeric_values

    return average_occurrence_of_zero

def compute_faithfulness(experiments: list[Type[NarrativeExperiment]])->list[Type[ExperimentMetrics]]:
    
    """Takes a list of experiments and computes the corresponding faithfulness metrics (RA/SA/VA)
    Args:
        experiments (list): A list of NarrativeExperiment objects containing the narratives and their parameters.
                            The objects in the list can also be ExperimentMetrics instances, but then these metrics will be overwritten.
    
    Returns:
        metrics (list): A list of ExperimentMetrics objects, where the RA/SA/VA have been added as attributes to the dataclass.
    """

    #initiate some variables that will be used later on
    num_feat=10 #dummy variable that should be larger than our truncated table
    rank_cols=[f"rank_{i}" for i in range(num_feat)]
    sign_cols=[f"sign_{i}" for i in range(num_feat)]
    value_cols=[f"value_{i}" for i in range(num_feat)]
    metrics=[]

    #loop over all experiments  
    for experiment in experiments:

        #initiate dictionaries where we will be saving the metrics
        rank_diff_dict={}
        sign_diff_dict={}
        value_diff_dict={}
        rank_accuracy_dict={}
        sign_accuracy_dict={}
        value_accuracy_dict={}

        #for each experiment loop over every extraction model and compute the accuracies for rank, sign and value
        for extraction_model, extractions_list in experiment.extractions_dict.items():
            
            
            rank_array=[]
            sign_array=[]
            value_array=[]

            #for a fixed extraction model loop over the individual extractions for each story

            for extraction, explanation in zip(extractions_list, experiment.explanation_list):
                
                #get the R/S/V differences of the extracted values with the real ones
                rank_diff, sign_diff, value_diff, _, _=ExtractionModel.get_diff(extraction,explanation)
                
                #for display purposes it can be useful to pad them with nans
                rank_diff+=[np.nan]*(num_feat-len(rank_diff))
                sign_diff+=[np.nan]*(num_feat-len(sign_diff))
                value_diff+=[np.nan]*(num_feat-len(value_diff))

                rank_array.append(rank_diff)
                sign_array.append(sign_diff)
                value_array.append(value_diff)

            #now we create full dataframes that contain every rank/sign/value error for every feature/narrative
            #remember the legend -- 0: identical, numerical but not 0: error, np.nan : to be ignored, np.inf: hallucinated feature

            rank_diff_df=pd.DataFrame(rank_array, columns=rank_cols)
            sign_diff_df=pd.DataFrame(sign_array, columns=sign_cols)
            value_diff_df=pd.DataFrame(value_array, columns=value_cols)

            #save these dataframes for every extraction model
            rank_diff_dict[extraction_model]=rank_diff_df
            sign_diff_dict[extraction_model]=sign_diff_df
            value_diff_dict[extraction_model]=value_diff_df

            #finally we compute the accuracy for every metric and save them
            rank_accuracy_dict[extraction_model]=average_zero(rank_diff_df[rank_cols[0:experiment.num_feat]])
            sign_accuracy_dict[extraction_model]=average_zero(sign_diff_df[sign_cols[0:experiment.num_feat]])
            value_accuracy_dict[extraction_model]= average_zero(value_diff_df[value_cols[0:experiment.num_feat]])
                
        metric=ExperimentMetrics(
                                    **asdict(experiment), 
                                    rank_diff=rank_diff_dict, 
                                    sign_diff=sign_diff_dict ,
                                    value_diff=value_diff_dict,
                                    rank_accuracy=rank_accuracy_dict,
                                    sign_accuracy=sign_accuracy_dict, 
                                    value_accuracy=value_accuracy_dict
        )
        metrics.append(metric)

    return metrics

 ##So in short -- rank_diff, sign_diff, value_diff dictionaries contain full dataframes with all errors for these quantities for every extraction model
 ## and rank/sign/value_accuracy dictionaries contain already averaged metrics for the entire experiment