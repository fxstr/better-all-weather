from pathlib import Path
from os import path
import pandas as pd

def get_data(instruments):
  """
  Returns local futures data for the instruments provided
  """
  current_dir = Path.cwd()
  relative_instrument_path = '../../data_futures_premium_data/data/'
  instrument_paths = [path.join(current_dir, relative_instrument_path, f'{instrument}.csv') for instrument in instruments]

  close_dfs = []
  for index, file in enumerate(instrument_paths):
      df = pd.read_csv(file, parse_dates=True, index_col='Date')
      # This is a Series
      close_value = df['Close']
      close_value.rename(instruments[index], inplace=True)
      close_dfs.append(close_value)

  closes = pd.concat(close_dfs, axis=1)
  closes.dropna(inplace=True)
  return closes
