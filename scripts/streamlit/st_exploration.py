import pickle
import streamlit as st
import pandas as pd
import os
import base64
from shapnarrative_metrics.llm_tools.extraction import ExtractionModel
from shapnarrative_metrics.experiment_management.experiment_dataclass import ExperimentMetrics
import emoji
import dill
from typing import Tuple, Type
import argparse
from scripts.figmakers.heatmap_ranksign import dfs_to_heatmap

METRICS_PATH=f"results/longshort_experiments/experiment_1/metrics.pkl"


if __name__=="__main__":

    with open(METRICS_PATH, "rb") as f:
        experiments: list[Type[ExperimentMetrics]]=dill.load(f)

    #some emojis"
    earth_emoji=emoji.emojize(':earth_americas:')
    dart_emoji=emoji.emojize(':dart:')
    scroll_emoji=emoji.emojize(':scroll:')
    thinking_emoji=emoji.emojize(':thinking_face:')
    barchart_emoji=emoji.emojize(':bar_chart:')
    tropy_emoji=emoji.emojize(':trophy:')
    brain_emoji=emoji.emojize(':brain:')

    highlight_color = 'background-color: #e0f7fa'

    # Function to highlight the highest values in specified columns
    def highlight_max(s):
        is_max = s == s.max()
        return [highlight_color if v else '' for v in is_max]

    # Function to highlight the lowest values in specified columns
    def highlight_min(s):
        is_min = s == s.min()
        return [highlight_color if v else '' for v in is_min]
    def highlight_second_min(s):
        sorted_unique_values = s.unique()
        sorted_unique_values.sort()
        if len(sorted_unique_values) > 1:
            second_min = sorted_unique_values[1]
        else:
            second_min = sorted_unique_values[0]
        is_second_min = s == second_min
        return [highlight_color if v else '' for v in is_second_min]

    def filter_experiments(id: Tuple, experiments: list) -> Type[ExperimentMetrics]:

        
        filtered_experiments=[experiment for experiment in experiments if experiment.id==id]

        if len(filtered_experiments)!=1:
            print(f"Something wrong length of filtered experiments is {len(filtered_experiments)}")
        
        return filtered_experiments[0]


    mode = st.sidebar.radio("Select Mode", ("Metrics Overview", "Metrics Figures", "Details"))

    #Generate all possible combinations for the experiment ids
    datasets=list(set(experiment.dataset for experiment in experiments))
    generation_models=list(set([ experiment.generation_model for experiment in experiments]))
    prompts=list(set(experiment.prompt_type for experiment in experiments))
    model_types=list(set(experiment.tar_model_name for experiment in experiments))

    # Sidebar
    dataset = st.sidebar.selectbox("Select Dataset", list(datasets))
    generation_model = st.sidebar.selectbox("Select Model (generation)", list(generation_models))
    prompt = st.sidebar.selectbox("Select Prompt Type (generation)", list(prompts))
    tar_model= st.sidebar.selectbox("Select Target Model Type", list(model_types))

    #Experiment id
    id=(dataset,tar_model,generation_model, prompt)
    human_exp=[experiment for experiment in experiments if experiment.generation_model=="human"]
    filtered_experiment=filter_experiments(id,experiments)

    if mode =="Metrics Figures":

        extraction_models=list(set( [ key for experiment in experiments for key in filtered_experiment.extractions_dict.keys()] ))
        embedding_models=list(set( [ key for experiment in experiments for key in filtered_experiment.embeddings_dict.keys()] ))
        extr_model = st.sidebar.selectbox("Select Extraction Model", list(extraction_models))
        emb_model= st.sidebar.selectbox("Select Embedding Model Type", list(embedding_models))

        st.write(f"**SHAP fidelity** ", "\n")
        fig_ranksign=dfs_to_heatmap(filtered_experiment.rank_diff[extr_model], filtered_experiment.sign_diff[extr_model],filtered_experiment.value_diff[extr_model])
        st.pyplot(fig_ranksign)

    elif mode=="Details":

        #Lower level selection within an experiment id
        extraction_models=list(set( [ key for experiment in experiments for key in filtered_experiment.extractions_dict.keys()] ))
        embedding_models=list(set( [ key for experiment in experiments for key in filtered_experiment.embeddings_dict.keys()] ))
        extr_model = st.sidebar.selectbox("Select Extraction Model", list(extraction_models))
        emb_model= st.sidebar.selectbox("Select Target Model Type", list(embedding_models))

        #Index slider
        selected_index = st.sidebar.slider("Select Index", 0, len(filtered_experiment.narrative_list) - 1)

        #Prep 
        print(filtered_experiment.generation_time)
        narrative=filtered_experiment.narrative_list[selected_index]
        explanation=filtered_experiment.explanation_list[selected_index]
        result=filtered_experiment.results_list[selected_index]
        test_idx=filtered_experiment.idx_list[selected_index]
        extracted_dict=filtered_experiment.extractions_dict[extr_model][selected_index]

        # Display selected narrative and instance table
        st.write(f"**Narrative** {scroll_emoji}", "\n")
        st.write(narrative, "\n")
        st.write(f"**Explanation (idx: {test_idx})** {barchart_emoji}", "\n")
        st.write(explanation)
        st.write(f"**Result** {barchart_emoji}", "\n")
        st.write(result)
        st.write(f"**Extraction** {tropy_emoji}")
        st.write(extracted_dict)
        st.write(f"**Rank difference**" )
        st.write(filtered_experiment.rank_diff[extr_model].iloc[selected_index])
        st.write("**Sign difference**" )
        st.write(filtered_experiment.sign_diff[extr_model].iloc[selected_index])

