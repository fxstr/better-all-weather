import pandas as pd

def split_data(data, interval='month', min_segment_duration=pd.Timedelta(days=365)):
  """
  Splits a dataframe into smaller timeframes whose distance is defined by interval that must be
  an attribute of a Pandas date (e.g. 'month', 'year'). Looks back for the given lookback period
  and returns a list of dataframes.
  """
  # Go thorugh all data and remember the ones that are in a new period (according to interval,
  # e.g. a new month)
  segments = []
  for i in range(1, len(data.index)):
    date = data.index[i]
    # Make sure that the first segment returned covers an interval that is longer than
    # min_segment_duration 
    if date - data.index[0] < min_segment_duration:
        continue
    if getattr(date, interval) != getattr(data.index[i-1], interval):
      segments.append(date)

  dfs = []
  for month_start_date in segments:
    series = data.loc[data.index[0]:month_start_date]
    dfs.append(series)

  return dfs
