import yaml
import pandas as pd 
import json
import numpy as np
import torch
from transformers import (AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline)
import os
import argparse
import csv

"""This script starts from the manipulated assumptions file that should have two columns (first column real assumptions, second column extracted) and computes the perplexity for both and saves the results to a json file and then saves a CSV that can be used directly for the figure in the paper. Requires llama 8 b so perhaps best suitable for cloud"""

#read excel file 
df_manip = pd.read_excel("data/manipulated_assumptions.xlsx")
model_name="meta-llama/Meta-Llama-3-8B"

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Compute perplexity for the extracted assumptions and their manipulated versions")
    parser.add_argument("--HF_token", default="", type=str, help=f"huggingface token for lama-3 8b base")
    args=parser.parse_args()
    HF_token=args.HF_token

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
        token=HF_token
    )

    tokenizer=AutoTokenizer.from_pretrained(model_name,
                                            token=HF_token)

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

    ppl_real=[]
    ppl_manip=[]
    for i in range(len(df_manip)):
      print(f"Computed perplexity {i}/{len(df_manip)}")
      ppl_real.append(perplexity(df_manip.iloc[i]["real_assumption"]))
      ppl_manip.append(perplexity(df_manip.iloc[i]["manipulated_assumption"]))
    
    dict_manip_assumptions={"real": ppl_real, "manip": ppl_manip}

    with open("results/figures/metrics_validation/manipulated_assumptions_ppl.json","w") as f:
        json.dump(dict_manip_assumptions, f)

    real=np.array(dict_manip_assumptions["real"])
    manip=np.array(dict_manip_assumptions["manip"])

    #sort by difference value
    diff=manip-real
    i_min=np.argmin(diff)
    i_max=np.argmax(diff)
    diff=np.sort(diff)

    #convert into csv for ease of use in tikz
    # Open or create a CSV file

    with open('results/figures/metrics_validation/manipulated_assumptions_ppl.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write rows of the two lists as columns
        for i in range(len(diff)):
            writer.writerow([diff[i] ])
