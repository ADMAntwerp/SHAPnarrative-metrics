import yaml
import pandas as pd 
import json
import numpy as np
import torch
from transformers import (AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline)
import os
import argparse
import dill
import copy

"""This script adds the perplexity metric to the extracted assumptions from our narrative experiments. 
   Should be run at any point after the human narratives were added since their assumptions need a perplexity too.
   We execute it on cloud since even llama 8b is RAM heavy. 
"""


def compute_perplexity(metrics, models, hf_token):

    """Computes the average perplexity of the extracted assumptions
    Args:
        metrics (list): list of ExperimentMetrics objects to which we will add the perplexity dictionaries
        models (list): list of HF-compatible model names relative to which we will compute the perplexities
        hf_tokes (str): HuggingFace token which has license for these models
    Returns:
        metrics (list): same metrics list as input, but now with the perplexities added
    """
    
    for metric in metrics:

        metric.perplexity_dict={model_name: {} for model_name in models}
        metric.perplexity_mean={model_name: {} for model_name in models}

    for model_idx, model_name in enumerate(models):

        bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model= AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=hf_token
        )

        tokenizer=AutoTokenizer.from_pretrained(model_name,
                                                token=hf_token)

        def perplexity(input_text):

            inputs = tokenizer(input_text, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
            loss
            # Compute perplexity
            perplexity = torch.exp(loss).item()

            return perplexity
        

        for metric in metrics:
            

            perplexity_dict= copy.deepcopy(metric.extractions_dict)
            perplexity_mean= copy.deepcopy(metric.extractions_dict)

            for extr_model, extr_list in metric.extractions_dict.items():
                ppl_list=[]
                for i,el in enumerate(extr_list):
                    print(f"extracting narrative {i} for dataset: {metric.dataset}")
                    for feature, extr in el.items():

                        if (extr["assumption"]=="None") or (extr["assumption"]==None):
                            ppl_val=np.nan
                        else:
                            ppl_val=perplexity(extr["assumption"])

                        perplexity_dict[extr_model][i][feature]=ppl_val
                        ppl_list.append(ppl_val)

                perplexity_mean[extr_model]=np.nanmean(ppl_list)

            metric.perplexity_dict[model_name]= perplexity_dict
            metric.perplexity_mean[model_name]= perplexity_mean   

    return metrics           

