import numpy as np
import pandas as pd


def full_inversion(explanation_df: pd.DataFrame, num_feat):

    """Manipulates a SHAP table and returns the manipulated version
    Arguments:
    -----------------
    explanation_df: pd.DataFrame
        A SHAP-table as created in self.gen_variables from the GenerationModel class
    num_feat: int
        Number of features to include in the truncated SHAP table
    Returns:
    -----------------
    manipulated_table: pd.DataFrame
        A truncated SHAP table with the ranks and signs inverted
    """

    explanation_df_manip=explanation_df.copy()

    #Create array of SHAP values that is inverted
    shuffled_SHAP =explanation_df.loc[0:num_feat-1]["SHAP_value"].iloc[::-1].values
    
    #Now add replace the old SHAP values in the table by the new inverted SHAP values, and also add an additional minus sign if the new SHAP value did not change sign:
    explanation_df_manip.loc[0:num_feat-1, "SHAP_value"] = [
    -val if (val * original_val > 0) else val
    for val, original_val in zip(shuffled_SHAP, explanation_df.loc[0:num_feat-1, "SHAP_value"])
    ]

    explanation_df_manip=explanation_df_manip.loc[explanation_df_manip["SHAP_value"].map(lambda x: np.abs(x)).sort_values(ascending=False).index]
    explanation_df_manip=explanation_df_manip.loc[explanation_df_manip["SHAP_value"].map(lambda x: np.abs(x)).sort_values(ascending=False).index]

    return explanation_df_manip

def shap_permutation(explanation_df: pd.DataFrame, num_feat):
    pass