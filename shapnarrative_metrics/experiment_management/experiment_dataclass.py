from dataclasses import dataclass, field
import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator
from typing import Tuple


@dataclass
class NarrativeExperiment:

    """This class contains data of one experiment where an experiment is defined as 
       one single list of narratives for a fixed data/prompt/model type with the corresponding embeddings and extractions in dicts"""

    #Misc params
    dataset: str
    prompt_type: str
    tar_model_name: str
    num_feat: int

    #Instance info
    idx_list: list[str]
    explanation_list: list[pd.DataFrame]
    results_list: list[pd.DataFrame]

    #Narrative generation
    narrative_list: list[str]
    generation_model: str

    #Rank/sign extractions for all extraction models
    extractions_dict: dict[list[dict]]

    #Embedding generation for all embedding models
    embeddings_dict: dict[list[np.array]]
    
    #Meta properties
    generation_time: float
    
    #identifier
    id: Tuple

@dataclass
class ExperimentMetrics(NarrativeExperiment):

    """This class inherits from an experiment dataclass and contains additional metrics"""
   
    #faithfulness
    rank_diff: dict= field(default_factory=dict)
    sign_diff: dict= field(default_factory=dict)
    value_diff:dict= field(default_factory=dict)
    rank_accuracy: dict= field(default_factory=dict)
    sign_accuracy: dict= field(default_factory=dict)
    value_accuracy: dict= field(default_factory=dict)

    #human similarity
    embedding_distance: dict = field(default_factory=dict)
    bleurt: dict = field(default_factory=dict)

    #assumptions
    perplexity_mean: dict = field(default_factory=dict)
    perplexity_dict: dict = field(default_factory=dict)
