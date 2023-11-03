import numpy as np
import pandas as pd

def remove_outlier(
    values: pd.Series or list,
    factor: float = 1.5,
    minimum_percentile: float = 0.25,
    maximum_percentile: float = 0.75
):
    first_quartile, third_quartile = np.percentile(
        values, [100*minimum_percentile, 100*maximum_percentile])
    quartilical_interval = third_quartile - first_quartile
    lowpass = first_quartile - (quartilical_interval * factor)
    highpass = third_quartile + (quartilical_interval * factor)
    values = values[(values >= lowpass) & (values <= highpass)]
    return values

def remove_outlier_dataframe(
    df: pd.DataFrame,
    column:str,
    factor: float = 1.5,
    minimum_percentile: float = 0.25,
    maximum_percentile: float = 0.75
):
    df = df.dropna()
    values = df[[column]]
    first_quartile, third_quartile = np.percentile(
        values, [100*minimum_percentile, 100*maximum_percentile])
    quartilical_interval = third_quartile - first_quartile
    lowpass = first_quartile - (quartilical_interval * factor)
    highpass = third_quartile + (quartilical_interval * factor)
    values = values[(values >= lowpass) & (values <= highpass)]
    df = df.drop(columns=column)
    df = pd.merge(df,values, left_index=True, right_index=True)
    df = df.dropna()
    return df