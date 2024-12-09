import pickle
import dill
import random
import pandas as pd 
from scipy.spatial import distance
import seaborn as sns
import numpy as np

"""In this script we generate all kinds of csv tables based on the experiments, the data from which is then combined in the main tables in the paper. 

To clarify the notation we will be dealing with several objects:

1) metric "object" (ExperimentMetrics class containing both the experiment params and the metrics measured for that narrative)

2) metrics_list "list" -- list of the metric objects that correspond to a single iteration over different datasets/llm_models. What you get when you do run_experiments.py in the scripts only once

3) metrics_iter "list of lists" where we have multiple metric_list in another lists that represents several identical iterations that allows us to account for remaining zero-temperature fluctuations
"""


file_name="metrics.pkl"
experiments_dir="standard_experiments"
standard_paths_iter=[f"results/{experiments_dir}/experiment_{i}/{file_name}" for i in range(1,5)]
experiments_dir="manipulated_experiments"
manip_paths_iter=[f"results/{experiments_dir}/experiment_{i}/{file_name}" for i in range(1,5)]
experiments_dir="longshort_experiments"
longshort_paths_iter=[f"results/{experiments_dir}/experiment_{i}/{file_name}" for i in range(1,2)]

def read_experiments(paths: list):

    """Function that reads in the experiment from a list of paths and combines them in one list"""
    metrics_iter=[]

    for path in paths:
        with open(path, "rb") as f:
            metrics_list=dill.load(f)
        metrics_iter.append(metrics_list)
    
    return metrics_iter

standard_metrics_iter=read_experiments(standard_paths_iter)
manip_metrics_iter=read_experiments(manip_paths_iter)
longshort_paths_iter=read_experiments(longshort_paths_iter)


def get_metric_tables(metrics_list, faithfulness_only=False):

    """Function that takes one metrics_list (list of metric objects) and then for every dataset generates a table for the different model performances and their averaged metrics for that dataset over the narratives"""

    datasets=["fifa","student","credit"]
    metrics_tables={key:pd.DataFrame( ) for key in datasets}

    for experiment in metrics_list:

        data=experiment.dataset
        model=experiment.generation_model
        prompt=experiment.prompt_type
        rank_accuracy=experiment.rank_accuracy.values()
        sign_accuracy=experiment.sign_accuracy.values() 
        value_accuracy=experiment.value_accuracy.values() 
        if faithfulness_only is False:
            hum_dist=experiment.embedding_distance.values() 
            ppl_llama=experiment.perplexity_mean["meta-llama/Meta-Llama-3-8B"].values() 
            ppl_mistral=experiment.perplexity_mean["mistralai/Mistral-7B-v0.3"].values() 
            bleurt=np.mean(experiment.bleurt)
        else:
            hum_dist=[np.nan]*20
            ppl_llama=np.nan
            ppl_mistral=np.nan
            bleurt=np.nan

        #reads previous table for that dataset
        table=metrics_tables[data]
        #concats with results for new model
        table = pd.concat([table, pd.DataFrame({'g-model': model, 'prompt': prompt, 'rank acc': rank_accuracy, "sign acc": sign_accuracy, "value acc":value_accuracy ,"hum sim": 1-list(hum_dist)[0], "bleurt":bleurt,"ppl_llama":ppl_llama, "ppl_mistral":ppl_mistral})], ignore_index=True)
        table.replace( {"claude-3-5-sonnet-20240620":"claude-3.5-sonnet","llama-3-70b-instruct":"llama-3-70b"},inplace=True)

        #appends to dataset again
        metrics_tables[data]=table
    
    return metrics_tables

def get_multiple_metric_tables(metrics_iter, faithfulness_only=False):

    """Generates metric tables across the _iter list over various fluctuations"""

    metrics_tables_iter=[]
    for metrics_list in metrics_iter:
        metrics_tables=get_metric_tables(metrics_list,faithfulness_only)
        metrics_tables_iter.append(metrics_tables)

    return metrics_tables_iter



numeric_colums=["rank acc","sign acc","value acc","hum sim","bleurt","ppl_llama","ppl_mistral"]

def get_average_over_ds(metric_tables_iter: list[dict])->pd.DataFrame:

    """Given an iteration list of metric tables as dictionaries, averages each metric table dictionary over the datasets"""

    averaged_df_iter=[]
    for metric_table in metric_tables_iter:
        averaged_df = metric_table["student"].copy()
        averaged_df[numeric_colums] = (metric_table["fifa"][numeric_colums] + metric_table["student"][numeric_colums] + metric_table["credit"][numeric_colums]) / 3
        averaged_df_iter.append(averaged_df)

    return averaged_df_iter


def get_minmax_from_iter(metric_tables_iter: list[pd.DataFrame]):

    """Given an iteration list of metric tables, computes the min and max of every element over all the tables in the list. Only makes sense to use after averaging over the datasets."""
    
    min_df=metric_tables_iter[0].copy()
    max_df=metric_tables_iter[0].copy()
    average_df=metric_tables_iter[0].copy()

    for i in range(len(metric_tables_iter)):
        for col in numeric_colums:
            min_df.loc[i,col]=np.min([df.loc[i,col] for df in metric_tables_iter])
            max_df.loc[i,col]=np.max([df.loc[i,col] for df in metric_tables_iter])
            average_df.loc[i,col]=np.mean([df.loc[i,col] for df in metric_tables_iter])

    return min_df, max_df, average_df


def get_diff(numeric_colums, df_standard, df_manip):
    assert (df_manip["g-model"]==df_standard["g-model"]).all()
    df=  df_manip[numeric_colums]-df_standard[numeric_colums]
    diff_df=pd.concat([df_standard[[col for col in df_standard.columns if col not in numeric_colums]], df], axis=1)
    return diff_df


if __name__=="__main__":

    """Now that the functions are predefined here comes the actual script for computing the tables"""
 
    ###LONGSHORT TABLE CSV
    longshort_tables_iter=get_multiple_metric_tables(longshort_paths_iter, faithfulness_only=True)
    average_longshort_table=get_average_over_ds(longshort_tables_iter)[0]
    average_longshort_table.to_csv("results/figures/tables/longshort/longshort_table.csv", index=False, float_format='%.3f')

    ###MAIN TABLE IN PAPER
    standard_tables_iter=get_multiple_metric_tables(standard_metrics_iter, faithfulness_only=False)
    manip_tables_iter=get_multiple_metric_tables(manip_metrics_iter, faithfulness_only=False)

    average_standard_tables_iter=get_average_over_ds(standard_tables_iter)
    average_manip_tables_iter=get_average_over_ds(manip_tables_iter)

    min_table_standard, max_table_standard, _=get_minmax_from_iter(average_standard_tables_iter)
    min_table_manip, max_table_manip, _=get_minmax_from_iter(average_manip_tables_iter)

    min_table_standard.to_csv("results/figures/tables/main/standard_table_min.csv", index=False, float_format='%.3f')
    max_table_standard.to_csv("results/figures/tables/main/standard_table_max.csv", index=False, float_format='%.3f')
    min_table_manip.to_csv("results/figures/tables/main/manip_table_min.csv", index=False, float_format='%.3f')
    max_table_manip.to_csv("results/figures/tables/main/manip_table_max.csv", index=False, float_format='%.3f')


    #also compute the differences
    diff_tables=[]
    for df_standard, df_manip in zip(average_standard_tables_iter,average_manip_tables_iter):
        diff_table=get_diff(numeric_colums, df_standard, df_manip)
        diff_tables.append(diff_table)

    min_difftable, max_difftable, _=get_minmax_from_iter(diff_tables)

    min_difftable.to_csv("results/figures/tables/main/diff_table_min.csv", index=False, float_format='%.3f')
    max_difftable.to_csv("results/figures/tables/main/diff_table_max.csv", index=False, float_format='%.3f')

    ###INDIVIDUAL DATASET TABLES
    print(standard_tables_iter[1])

    for ds_name in ["fifa","student","credit"]:

        ds_standard_tables_iter=[table[ds_name] for table in standard_tables_iter]
        ds_manip_tables_iter=[table[ds_name] for table in manip_tables_iter]
        _,_, average_table_standard=get_minmax_from_iter(ds_standard_tables_iter)
        _,_, average_table_manip=get_minmax_from_iter(ds_manip_tables_iter)
        difftable=get_diff(numeric_colums, average_table_standard, average_table_manip)

        average_table_standard.to_csv(f"results/figures/tables/individual/{ds_name}_standard_table.csv", index=False, float_format='%.3f')
        average_table_manip.to_csv(f"results/figures/tables/individual/{ds_name}_manip_table.csv", index=False, float_format='%.3f')
        difftable.to_csv(f"results/figures/tables/individual/{ds_name}_diff_table.csv", index=False, float_format='%.4f')



