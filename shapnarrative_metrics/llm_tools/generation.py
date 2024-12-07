from typing import Type
import numpy as np
import pandas as pd
import shap
from shapnarrative_metrics.llm_tools.llm_wrappers import LLMWrapper
import json
from shapnarrative_metrics.llm_tools.extraction import ExtractionModel
from shapnarrative_metrics.misc_tools.manipulations import full_inversion, shap_permutation

import random

class GenerationModel:

    """
    A class to generate SHAP narratives that explain AI predictions based on SHAP values.

    Attributes:
    -----------
    ds_info: dictionary
        Dictionary that contains short generic information about the dataset  
        Format: {
                 "dataset description": dataset_description (str), 
                 "target_description": target_description (str), 
                 "task_description": task_description (str), 
                 "feature_desc": pd.DataFrame that contains columns "feature_name", "feature_average", "feature_desc".
                 }
    llm: object
        Initialized LLM wrapper from the llm_wrappers module with an API call functionality
    """

    def __init__(
        self,
        ds_info:dict,
        llm: Type[LLMWrapper] = None
    ):
        
        self.ds_info=ds_info
        self.feature_desc=ds_info["feature_description"]
        
        if llm is None:
            print(
                "No language model provided. You will only be able to generate prompts."
            )
        else:
            self.llm = llm


    def gen_shap_feature_df(self, x, model, tree):


        """
        Generates a DataFrame with original features and their corresponding SHAP values.

        Parameters:
        -----------
        x : DataFrame
            The input data for which SHAP values are to be calculated.
        model : object
            A trained model which supports SHAP explanations.
        tree : bool, default=True
            Boolean indicating if the model is tree-based. If True, TreeExplainer will be
            used for SHAP explanations, else KernelExplainer will be used.
        Returns:
        --------
        DataFrame
            A DataFrame with original features and their SHAP values.
        """

        if tree:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(x)[:, :, 1]
        else:
            explainer = shap.KernelExplainer(model.predict, x)
            shap_vals = explainer.shap_values(x).T

        shap_df = pd.DataFrame(shap_vals, columns=x.columns)

        return shap_df

    def gen_variables(self, model, x, y, tree):

        """
        Generate necessary variables including SHAP values and predictions.

        Parameters:
        -----------
        model : object
            A trained model which supports SHAP explanations.
        x : DataFrame
            The input data. Could essentially be "x_test". Important: columns should contain features in the same order as the feature_desc kwarg in ds_info. 
        y : Series or array-like
            The true labels for the input data.
        tree : bool, default=True
            Boolean indicating if the model is tree-based. If True, TreeExplainer will be
            used for SHAP explanations, else KernelExplainer will be used.
        """

        shap_vals = self.gen_shap_feature_df(x, model, tree)
        self.explanation_list = []

        # Compute a list of SHAP tables for every input datapoint, where a "SHAP table" is a dataframe containing combined information about the instance and explanation: feature_names, SHAP_value, feature_values, feature_average, feature_description.
        # Important: we decide to sort by absolute SHAP values in our study, should be changed if other choices are made
        for i in range(len(x)):
            explanation = self.feature_desc.copy()
            explanation.insert(1, "feature_value", x.iloc[i].to_numpy())
            explanation.insert(1, "SHAP_value", shap_vals.iloc[i].to_numpy())
            explanation["abs_SHAP"] = explanation["SHAP_value"].abs()
            explanation = explanation.sort_values(by="abs_SHAP", ascending=False)
            explanation.drop(columns=["abs_SHAP"], inplace=True)
            explanation=explanation.reset_index().drop(columns=["index"])
            self.explanation_list.append(explanation)

        # Generate predictions and probability scores for the input data.  
        y_pred = model.predict(x)
        y_scores = model.predict_proba(x)[:, 1]

        #Save the results as in a dataframe. 
        self.result_df = pd.DataFrame({"truth": y, "pred": y_pred, "score": y_scores})


    def generate_story_prompt(self, iloc_pos, num_feat=4 ,sentence_limit=10, prompt_type="long",manipulate=False, manipulation_func=full_inversion):
        

        """Generate a prompt for a given SHAP table
        Parameters:
            iloc_pos : (int) Which SHAP table in self.explanation_list to generate the prompt for
            num_feat: (int) Number of features to keep in the truncated SHAP table left in the prompt
            sentence_limit: (int) Ask LLM to limit answer to this sentence limit
            prompt_type: str (short/long) We consider two prompt templates, a long and a short one, select which one to generate.
            manipulate: bool (default=False) If you pass True, then the truncated SHAP table will be manipulated at the prompt level.
            manipulation_func: (default= full inversion) function(explanation_df, num_feat)->manipulated_explanation_df that returns a manipulated truncated table.

        Returns:
            prompt: (str) Prompt that is ready to be passed to an LLM
        """
        
        
        result_string=f"""The model predicted a probability of {self.result_df.score.iloc[iloc_pos]:.2%} for the target variable being 1,
                    and therefore predicted the outcome {self.result_df.pred.iloc[iloc_pos]}."""
        
        explanation_df=self.explanation_list[iloc_pos].copy()
        

        ### PROMPT MANIPULATION
        "If manipulate is passed, create a manipulated SHAP table that we will pass to the LLM"
        if manipulate==True:
            explanation_df=manipulation_func(explanation_df, num_feat)

        explanation_table=explanation_df.iloc[0:num_feat].to_string(index=False)
        

        #### PROMPTS 

        short_prompt=f"""
        Your goal is to generate a textual explanation or narrative as to why an AI model made a certain prediction for one particular instance.

        Here is the data:

        Dataset description: {self.ds_info["dataset_description"]},
        Target description: {self.ds_info["target_description"]},
        Task description": {self.ds_info["task_description"]},
        Feature table: {explanation_table},
        Result: {result_string}

        Please generate a fluent and cohesive narrative that explains the prediction made by the model. 

        You can start the answer immediately with the explanation. 
        Limit your answer to {sentence_limit} sentences and {num_feat} features.  
        Do not use tables or lists, or simply rattle through the features one by one. The goal is to have a narrative/story.
        """

        long_prompt=f"""

        Your goal is to generate a textual explanation or narrative as to why an AI model made a certain prediction for one particular instance. 
        To do this, you will be provided with a dictionary that contains broad descriptions of the specific dataset, target variable, and the task the model was trained on.
        Additionally, you will be provided with a dataframe that contains the names of all the features, their descriptions, their values, their average values and their SHAP values.
        Finally you will get a single string describing the result of the prediction.

        The goal of SHAP is to explain the prediction of an instance by computing the contribution of each feature to the prediction.
        Each individual SHAP value is a measure of how much additional probability this feature adds or subtracts 
        in the predicted probability relative to the base level probability. 
        This relative nature of the SHAP values might have unexpected consequences that you are to take into account.
        For example, features that should intuitively contribute in a positive way (and vice versa), 
        can still have negative SHAP values if their value is below an average in the dataset.

        Here is the data:

        Dataset description: {self.ds_info["dataset_description"]},
        Target description: {self.ds_info["target_description"]},
        Task description": {self.ds_info["task_description"]},
        Feature table: {explanation_table},
        Result: {result_string}

        The features in the "Feature table" above are already sorted by their absolute SHAP values to make it easier for you to locate the most important ones. 
        Always provide the probability of getting outcome 1, even if the predicted outcome is 0. 

        Please generate a fluent and cohesive narrative that explains the prediction made by the model. In your answer follow these rules.

        Format related rules:

        1) You can start the answer immediately with the explanation. 
        2) The answer should end with a single sentence summarizing and weighing the different factors. 
        3) Limit the entire answer to {sentence_limit} sentences or fewer. 
        4) Only invoke the {num_feat} most important features in the narrative. 
        5) Do not use tables or lists, or simply rattle through the features one by one. The goal is to have a narrative/story.

        Content related rules:

        1) Be clear about what the model actually predicted and how certain it was about that prediction. 
        2) Discuss how the features contributed to that prediction and be clear about whether a particular feature had a positive or negative contribution relative to the probability of getting outcome 1.
           Make sure to clearly establish this the first time you refer to a feature.
        3) Consider the relative magnitude of the absolute SHAP values of the features when referencing their relative importance.
        4) The reader should be able to tell what the order of importance of the features is based on their absolute SHAP values. 
        5) You should provide a suggestion or interpretation as to why a feature contributed in a certain direction. Try to introduce external knowledge that you might have that is not in the SHAP table.  
        6) If there is no simple explanation for the effect of a feature, try to consider the context of other features in the interpretation.
        7) Do not use the SHAP numeric values in your answer.
        8) You can use the feature values themselves in the explanation, as long as they are not categorical variables.
        9) Do not use the comparison with the average for every single feature, and reserve it for features where it really clarifies the explanation.  
        """

        prompt_dict={
                     "short":short_prompt,
                     "long":long_prompt
                     }

        return prompt_dict[prompt_type]



    def generate_stories(self, model, x, y, tree=True, prompt_type="long", num_feat=4, manipulate=False, manipulation_func=full_inversion):
        """
        Generates SHAPstories for each instance in the given data.

        Parameters:
            model : (object) A trained model which supports SHAP explanations.
            x : (DataFrame) The input data.
            y : (Series or array-like). The true labels for the input data.
            tree : (bool) (default=True)Boolean indicating if the model is tree-based. If True, TreeExplainer will be
            used for SHAP explanations, else KernelExplainer will be used.
            num_feat: (int) Number of features to keep in the truncated SHAP table left in the prompt
            manipulate: (bool) (default=False) If you pass True, then the truncated SHAP table will be manipulated at the prompt level.
            manipulation_func: (func) function to start from explanation table and create a truncated manipulated table

        Returns:
            list of str: A list containing the generated narratives for each instance.
        """

        self.gen_variables(model, x, y, tree)
        stories = []

        for i in range(len(x)):

            story_prompt=self.generate_story_prompt(i, prompt_type=prompt_type, num_feat=num_feat, manipulate=manipulate, manipulation_func=manipulation_func)

            story=self.llm.generate_response(story_prompt) 

            stories.append(story)
            print(f"Generated story {i+1}/{len(x)} with {self.llm.model}")

        return stories
