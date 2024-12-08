import argparse
import yaml
import pandas as pd
from shapnarrative_metrics.llm_tools.llm_wrappers import GptApi, LLMWrapper, LlamaAPI, ClaudeApi, MistralApi
from shapnarrative_metrics.llm_tools.generation import GenerationModel
from shapnarrative_metrics.experiment_management.experiment_manager import ExperimentManager
from shapnarrative_metrics.experiment_management.experiment_dataclass import NarrativeExperiment
from shapnarrative_metrics.llm_tools.embedding_wrappers import EmbedWrapper, VoyageEmbedder
from shapnarrative_metrics.misc_tools.manipulations import full_inversion, shap_permutation
import json
import numpy as np
import pickle
import dill
from dataclasses import dataclass
from typing import Tuple


"""THIS IS THE MAIN SCRIPT TO GENERATE NARRATIVE TOGETHER WITH EXTRACTIONS FOR ALL TYPES OF PARAMETERS AND COMBIANTIONS
   DEPENDING ON THE SPECIFIC EXPERIMENT MANUALLY CHANGE PATHS, MANIPULATE BOOL and MANIPULATE FUNC etc"""

#PATHS
HUMAN_NARRATIVE_PATHS="data/human_written.json"
TEMP_SAVE_PATH="results/temp/latest_experiment.pkl"
# SAVE_PATHS=[f"results/standard_experiments/experiment_{i}/experiment.pkl" for i in range(2,5)]
# SAVE_PATHS=[f"results/longshort_experiments/experiment_{i}/experiment.pkl" for i in range(1,2)]
# SAVE_PATHS=[f"results/manipulated_experiments/experiment_{i}/experiment.pkl" for i in range(1,5)]
SAVE_PATHS=[f"results/manipulated_experiments/permutation_manip/experiment.pkl"]
#DEFAULT PARAMS
SIZE_LIMIT=20
# GEN_MODEL_LIST=["gpt-4o","llama-3-70b-instruct","claude-sonnet-3.5","mistral-large-2407"]
GEN_MODEL_LIST=["gpt-4o"]
EXT_MODEL_LIST=["gpt-4o"]
EMB_MODEL_LIST=["voyage-large-2-instruct"]
DATASET_NAMES=["fifa","credit","student"]
TAR_MODEL_NAMES=["RF"]
PROMPT_TYPES=["long"]
# PROMPT_TYPES=["long","short"]

NUM_FEAT=4
MANIPULATE=1
# MANIPULATE_FUNC=full_inversion
MANIPULATE_FUNC=shap_permutation
APPEND_HUMAN=True

#LOAD CONFIG:
with open("config/keys.yaml", "r") as file:
    config_data = yaml.safe_load(file)
OpenAI_API_key = config_data["API_keys"]["OpenAI"]
Replicate_API_key = config_data["API_keys"]["Replicate"]
Voyage_API_key = config_data["API_keys"]["Voyage"]
anthropic_key=config_data["API_keys"]["Anthropic"]
mistral_key=config_data["API_keys"]["Mistral"]


#LLM PARAMS
SYSTEM_ROLE="You are a teacher that explains AI predictions"
SYSTEM_ROLE_EXTRACTION="You are an analyst of few words that extracts subtle clues from text."
GENERATION_TEMPERATURE=0
EXTRACTION_TEMPERATURE=0

#LLM Options
LLM_WRAPPERS={ 
               "llama-3-70b-instruct": LlamaAPI(Replicate_API_key, model="llama-3-70b-instruct", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
               "gpt-4o": GptApi(OpenAI_API_key,model="gpt-4o", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
                "claude-sonnet-3.5": ClaudeApi(anthropic_key,model="claude-3-5-sonnet-20240620", system_role=SYSTEM_ROLE, temperature=GENERATION_TEMPERATURE),
                "mistral-large-2407": MistralApi(api_key=mistral_key, model="mistral-large-2407" ,system_role="You are a teacher that explains AI predictions.", temperature=GENERATION_TEMPERATURE)
               }
EXT_WRAPPERS={
               "gpt-4o": GptApi(OpenAI_API_key,model="gpt-4o", system_role=SYSTEM_ROLE_EXTRACTION, temperature=EXTRACTION_TEMPERATURE)
               }
EMB_WRAPPERS={
               "voyage-large-2-instruct": VoyageEmbedder(api_key=Voyage_API_key, model="voyage-large-2-instruct")
               }
DS_PATHS={
    "fifa": "data/fifa_dataset",
    "credit": "data/credit_dataset",
    "student": "data/student_dataset",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a series of narratives and save them")
    parser.add_argument("--SAVE_PATHS", '--save_paths_list' , nargs='+', default=SAVE_PATHS, type=list, help=f"Path to dir where the narratives will be saved")
    parser.add_argument("--NUM_FEAT", default=NUM_FEAT, type=int, help=f"Number of features to use in narrative")
    parser.add_argument("--DATASET_NAMES", '--dataset_names_list', nargs='+', default=DATASET_NAMES,  help=f"Names of datasets in experiment")
    parser.add_argument("--TAR_MODEL_NAMES", '--target_names_list', nargs='+', default=TAR_MODEL_NAMES,  help=f"Names of target models")
    parser.add_argument("--GEN_MODEL_LIST", '--gen_model_names_list', nargs='+', default=GEN_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {LLM_WRAPPERS.keys()}")
    parser.add_argument("--EXT_MODEL_LIST", '--ext_model_names_list', nargs='+', default=EXT_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {EXT_WRAPPERS.keys()}")
    parser.add_argument("--EMB_MODEL_LIST", '--emb_model_names_list', nargs='+', default=EMB_MODEL_LIST,  help=f"Can be one or multiple models chosen from: {EMB_WRAPPERS.keys()}")
    parser.add_argument("--PROMPT_TYPES", '--prompt_types_list', nargs='+', default=PROMPT_TYPES,  help=f"List of types for prompt experiment")
    parser.add_argument("--SIZE_LIMIT", type=int, default=SIZE_LIMIT,  help=f"Max number of samples to be taken from test set")
    parser.add_argument("--WARM_START", type=int, default=0,  help=f"Warm start for experiments from existing temp dict")
    parser.add_argument("--MANIPULATE", type=int, default=MANIPULATE,  help=f"Manipulate experiments or not")

    args = parser.parse_args()

    gen_models=[LLM_WRAPPERS[model_name] for model_name in args.GEN_MODEL_LIST]
    ext_models=[EXT_WRAPPERS[model_name] for model_name in args.EXT_MODEL_LIST]
    emb_models=[EMB_WRAPPERS[model_name] for model_name in args.EMB_MODEL_LIST]
    
    print(f"MANIP = {args.MANIPULATE}")

    for save_path in args.SAVE_PATHS:

        manager=ExperimentManager(
            dataset_names=args.DATASET_NAMES,
            tar_model_names=args.TAR_MODEL_NAMES,
            generation_models=gen_models,
            extraction_models=ext_models,
            embedding_models=emb_models,
            prompt_types=args.PROMPT_TYPES,
            ds_paths=DS_PATHS,
            size_lim=args.SIZE_LIMIT,
            num_feat=args.NUM_FEAT
        )

        experiment_list=manager.run_experiments(temp_save_path=TEMP_SAVE_PATH, warm_start=bool(args.WARM_START),manipulate=bool(args.MANIPULATE), manipulation_func=MANIPULATE_FUNC)


        #ADD HUMAN WRITTEN NARRATIVES FROM DICT IN THE SAME FORMAT THE LLM EXPERIMENTS
        if APPEND_HUMAN is True:

            with open(HUMAN_NARRATIVE_PATHS,"r") as f:
                human_dict=json.load(f)
            experiment_list=manager.append_human(
                                                 experiments=experiment_list,
                                                 human_dict=human_dict,
                                                 ext_models=[EXT_WRAPPERS[ext_model_name] for ext_model_name in EXT_MODEL_LIST],
                                                 emb_models=[EMB_WRAPPERS[emb_model_name] for emb_model_name in EMB_MODEL_LIST]
                                                 )
            

        with open(save_path, "wb") as f:
            dill.dump(experiment_list, f)
    
