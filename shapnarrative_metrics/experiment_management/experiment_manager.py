import pandas as pd
import pickle
from shapnarrative_metrics.llm_tools.llm_wrappers import LLMWrapper
from shapnarrative_metrics.llm_tools.generation import GenerationModel
from typing import Type
from shapnarrative_metrics.llm_tools.extraction import ExtractionModel
from shapnarrative_metrics.experiment_management.experiment_dataclass import NarrativeExperiment
from shapnarrative_metrics.misc_tools.manipulations import full_inversion

import itertools
import time
from dataclasses import asdict


class ExperimentManager():

    """ 
    A class to manage generation of narratives across different parameters. Input variables represent all experiment combinations to perform.

    Attributes:
        dataset_names: (list) list of names of the datasets to be used
        dataset_names: (list) list of names for the target models to be used
        generation_models: (list) list of LLM wrapped generation models
        extraction_models: (list) list of LLM wrapped extraction models
        embedding_models: (list) list of embedding-wrapped embedding models
        prompt_types: (list) list of prompt types allowed in the generation model class
        ds_paths: (dict) dictionary with ds_names as keys, and the paths to that dataset location as values
        size_lim: (int) number of narratives to create for a given dataset
        num_feat: (int) number of features to use in t
    """

    def __init__(
         self,
         dataset_names: list[str],
         tar_model_names: list[str],
         generation_models: list[Type[LLMWrapper]],
         extraction_models: list[Type[LLMWrapper]],
         embedding_models: Type[LLMWrapper],
         prompt_types: list[str],
         ds_paths: dict,
         size_lim: int,
         num_feat: int
    ):
        
        self.dataset_names=dataset_names
        self.tar_model_names=tar_model_names
        self.generation_models=generation_models
        self.extraction_models=extraction_models
        self.embedding_models=embedding_models
        self.prompt_types=prompt_types
        self.ds_paths=ds_paths
        self.size_limit=size_lim
        self.num_feat=num_feat
  

    def sample_equal_targets(self, df):

        """Given a dataframe representing a large test set randomly samples n_tot instances from it such that it has an equal number of classes
        Args:
            df: (pd.DataFrame) Test_set containing the class label in a "target" variable

        Returns:
            sampled_df: (pd.DataFrame) sampled test set to be used for narrative generation 
        """
        

        target_name=df.columns[-1]
        df_target_0 = df[df[target_name] == 0]
        df_target_1 = df[df[target_name] == 1]

        # Randomly sample entries from each subset
        sample_0 = df_target_0.sample(int(self.size_limit/2), random_state=42)
        sample_1 = df_target_1.sample(int(self.size_limit/2), random_state=42)

        # Combine the samples
        sampled_df = pd.concat([sample_0, sample_1])

        # Shuffle the combined sample to mix the rows
        sampled_df = sampled_df.sample(frac=1, random_state=42) 

        return sampled_df

    def dataset_extraction_tool(self, dataset_name, tar_model_name):

        """Given a dataset name reads in all necessary data from the data folder that we need for narrative generation

        Args:
            dataset_name: name of dataset to read the path through the self.ds_paths dictonary
            tar_model_name: name of the target model for which the SHAP values will be generated (e.g. RF)

        Returns:
            sampled_test_set: output of the sample_equal_targets method
            ds_info: dictionary that was prepared during data preprocessing containing necassary info for prompt
            tar_model: read-in pickled target model
        """
        
        test_set=pd.read_parquet(f"{self.ds_paths[dataset_name]}/test_cleaned.parquet") 
        sampled_test_set=self.sample_equal_targets(test_set)

        with open(f"{self.ds_paths[dataset_name]}/dataset_info", 'rb') as f:
            ds_info= pickle.load(f)
        
        with open(f"{self.ds_paths[dataset_name]}/{tar_model_name}.pkl", 'rb') as f:
            tar_model= pickle.load(f)

        return sampled_test_set, ds_info, tar_model
    
    def run_experiments(
         self,
         temp_save_path: str = "results/temp/latest_experiment.pkl",
         warm_start: bool = False,
         manipulate: bool= False,
         manipulation_func =  full_inversion
        ):
    
        """Executes experiments in all possible combinations we are interested in for the study using the combinations of the parameters provided at the start of class initialization.
        args:
            temp_save_path: saves intermediate results in temporary dir, in case an api call fails to avoid redoing everything.
            warm_start: start from latest temp_save_path result?
            manipulate: bool -- manipulate narratives or not
        returns:
            list of NarrativeExperiment dataclasses as defined in the experiment_dataclass module
        """
        experiments=[]

        #in case an api call failed in the previous experiment run, we can start from the last saved:
        if warm_start:
            with open(temp_save_path, 'rb') as f:
                experiments=pickle.load(f)
            existing_ids=[exp.id for exp in experiments]

        #start iteration loop for all possible combination of experiment parameters 
        total_iterations = len(self.dataset_names) * len(self.tar_model_names) * len(self.generation_models)* len(self.prompt_types)  
        for index, (dataset, tar_model_name, gen_model, prompt_type) in enumerate(itertools.product(self.dataset_names,self.tar_model_names,self.generation_models, self.prompt_types)):

            #define unique experiment id tuple for the loop and in case of warm start check if it already exists
            id=(dataset,tar_model_name,gen_model.model, prompt_type)
            if warm_start:
                if id in existing_ids:
                    continue
                
            print(f"****** Running experiment {index+1}/{total_iterations} with {id} ******")
 
            #extract everything we need from the dataset directory
            test_set, ds_info, tar_model=self.dataset_extraction_tool(dataset, tar_model_name)
            
            #generate set of stories if this set has not yet been generated before
            story_generator=GenerationModel(ds_info, gen_model)

            start_time=time.time()
            narratives=story_generator.generate_stories(tar_model,test_set[test_set.columns[0:-1]],test_set[test_set.columns[-1]], prompt_type=prompt_type, num_feat=self.num_feat, manipulate=manipulate, manipulation_func=manipulation_func)
            end_time=time.time()
            generation_time=float(end_time-start_time)
            generation_time=generation_time/len(narratives)

            explanation_list=story_generator.explanation_list
            result_df=story_generator.result_df


            #generate extractions with every extraction model
            extractions_dict={}
            for extr_model in self.extraction_models:
                feature_extractor=ExtractionModel(ds_info, extr_model)
                extractions=feature_extractor.generate_extractions(narratives)
                extractions_dict[extr_model.model]=extractions
  
            #generate embeddings
            embeddings_dict={}
            for emb_model in self.embedding_models:
                embeddings=[emb_model.generate_embedding(narrative) for narrative in narratives]
                embeddings_dict[emb_model.model]=embeddings

            #store experiment data
            experiment=NarrativeExperiment(
                dataset= dataset,
                prompt_type=prompt_type,
                tar_model_name=tar_model_name,
                idx_list=list(result_df.index),
                explanation_list=explanation_list,
                results_list=[result_df.iloc[i] for i in range(len(result_df))],
                narrative_list=narratives,
                generation_model=gen_model.model,
                extractions_dict=extractions_dict,
                embeddings_dict=embeddings_dict,
                generation_time=generation_time,
                id=id,
                num_feat=self.num_feat
            )

            experiments.append(experiment)

            #save after every iteration in case any api call fails to be able to restart it
            with open(f"{temp_save_path}", 'wb') as f:
                pickle.dump(experiments,f)
        
        return experiments
    
    def append_human(self,experiments: list[Type[NarrativeExperiment]], human_dict: dict, ext_models: list[Type[LLMWrapper]], emb_models: list):

        """
        Given a list of experiments generated with LLMs with self.run_experiments and corresponding human written narratives, add the human narratives to the experiment as if it is just another generation model. 
        Assumes that the human written narratives are on exactly the same datasets and instances as the LLM generated ones

        Args:
            experiments (list): list of experiments generated with self.run_experiments
            human_dict (dict): dictionary with the human narratives in the format {$dataset_name: {narrative_list:[...], idx_list: [...]}} for all    datasets and 
            ext_models (list): list of wrapped extraction models to generate extractions for the human narratives
            emb_models (list): list of wrapped embedding models to generate embeddings for the human narratives
        Returns: 
            list of experiments where the human narratives have been added in the same format as if they are a generation model
        """

        manual_experiments=[]

        for ds_name, value in human_dict.items():
            
            #Get any experiment object that matches the current dataset
            matching_experiment=self.filter_experiments(ds_name, "long", "RF", experiments)

            #Create a now dataclass starting from the matching experiment and overwrite several attributes
            manual_experiment=NarrativeExperiment(**asdict(matching_experiment))
            manual_experiment.generation_model="human"
            manual_experiment.id=(ds_name,"RF","human","long")
            manual_narratives=human_dict[ds_name]["narrative_list"]

            #Make sure that nothing went wrong and that the indices match
            assert matching_experiment.idx_list==human_dict[ds_name]["idx_list"], "The append_human method assumes that the human and LLM indices match."

            #Get the correct ds_info object for this dataset
            _, ds_info, _ =self.dataset_extraction_tool(ds_name, "RF")


            #Generate extractions
            extractions_dict={}
            for extr_model in ext_models:
                feature_extractor=ExtractionModel(ds_info, extr_model)
                extractions=feature_extractor.generate_extractions(manual_narratives)
                extractions_dict[extr_model.model]=extractions

            manual_experiment.extractions_dict=extractions_dict

            #generate embeddings 
            embeddings_dict={}
            for emb_model in emb_models:
                embeddings=[emb_model.generate_embedding(narrative) for narrative in manual_narratives]
                embeddings_dict[emb_model.model]=embeddings

            #overwrite the narratives, extractions and embeddings with the human ones
            manual_experiment.embeddings_dict=embeddings_dict
            manual_experiment.narrative_list=manual_narratives
            manual_experiments.append(manual_experiment)

        #add manual experiments to existing experiment list and save
        experiments+=manual_experiments

        return experiments

    
    @staticmethod
    def filter_experiments(dataset: str, prompt:str, tar_model:str, experiments: list) -> Type[NarrativeExperiment]:

        """Filters a list of experiments on an experiment with a particular dataset/prompt/tar_model"""

        filtered_experiments=[experiment for experiment in experiments if (experiment.dataset==dataset) & (experiment.prompt_type==prompt) & (experiment.tar_model_name==tar_model)]

        return filtered_experiments[0]





 