from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

regression = LinearRegression()

def calculate_slope(series):
    # Gives a list of days that have passed since the first day, for every row 
    time_delta = (series.index - series.index[0]).to_series().dt.days
    y = series.values.reshape(-1, 1)
    x = time_delta.to_numpy().reshape(-1, 1)
    regression.fit(x, y)
    intercept = regression.intercept_[0]
    slope = regression.coef_[0][0]
    return ((intercept + slope * 365) / intercept) - 1

def get_relative_slope(series):
    '''
    Calculates the relative slope of a linear regression line for the given data. Returns a
    percent value (annualized expected return). Only returns values for the first (trading) day
    of the month to improve performance.
    Parameters
    ----------
    data  : pd.Series
        Series with index as datetime and values as floats; use a series to easily 
        call df.rolling().apply()

    Returns
    -------
    pd.Series
        Series with the relative slope for each instrument (index is the instrument, value the
        slope)
    '''
    # Not enough data to check for month changes: return
    if len(series) < 2:
        return np.nan
    # Last entry is not a new month: return NaN
    if series.index[-1].month == series.index[-2].month:
        return np.nan
    return calculate_slope(series)