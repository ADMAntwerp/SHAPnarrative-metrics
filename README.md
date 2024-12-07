## SHAPnarrative-metrics

In this repository we explore several automated metrics for XAI narratives based on SHAP. 

This is a research-level repository for reproducibility purposes accompanying our recent paper: ... 

A proper package for a practical implementation of metrics will be made available soon. [link to follow].


illustrative picture:
![Screenshot](path)


## Setup
Clone repository:

```python
git clone ...
````

Create venv and activate:
 
```python
python -m venv venv
source venv/bin/activate
```

Install your package in editable mode (required step for paths to work):

```python
pip install -e .
```

## Config keys

Create a `config/keys.yaml` file with API keys for: OpenAI, Anthropic, Replicate (used for Llama-3 70b), and Mistral. Also include a HuggingFace token with access to Llama-3 8b, and Mistral-7b. 

## Data prep

All data prep happens inside the notebooks inside `data/`, and the results are saved in the respective directories.

We did not aim to attain an optimal performance for the target model and hence the preprocessing is pretty basic. 

## 