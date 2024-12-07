from setuptools import find_packages, setup

packages = [
    "numpy",
    "pyyaml",
    "shap",
    "openai",
    "scikit-learn==1.4.2",
    "pandas",
    "transformers",
    "pyarrow",
    "streamlit",
    "anthropic",
    "replicate",
    "voyageai",
    "mistralai",
    "pmlb",
    "dill",
    "matplotlib",
    "seaborn",
    "torch",
    "evaluate"
]

setup(
    name="shapnarrative_metrics",
    version="0.0",
    description="Generation, extraction and metrics for XAI SHAP narratives",
    packages=find_packages(),
    install_requires=packages,
)
