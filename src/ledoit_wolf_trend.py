# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # What do we do here? 
# - Get data from Tiingo
# - Use linear regression slope of multiple timeframes to determine every instrument's trend.
#   Across every timeframe, we use -1 for a negative trend and +1 for a positive one to account for
#   the steeper slopes of shorter timeframes. We then use the mean across those -1/1 values to
#   determine the trend.
# - Pass all instrument with positive trends to the Ledoit-Wolf algorithm to determine the weights.
#   We use Ledoit Wolf (that simplifies covariance estimation) and optimize for maximum
#   diversification.
# - We average the trends; this results in a number between -1 and 1; we adjust it to go from 0 to 1 
#   which gies us the target exposure; we then multiply our weights by it to get the adjusted weights.
#
#   We could go weekly with trend [20, 30, 40, 50, 60] and 30 lookback for Ledoit-Wolf.
#
# ## Expectations
# - Triple ETFs & max. exp. = long / all: 12% CAGR, 29% MaxDD ➝ 2.42
# - Triple ETFs & max. exp. = 100%: 16% CAGR, 39% MaxDD ➝ 2.43
# - Double ETFs & max. exp. = 100%: 13% CAGR, 38% MaxDD ➝ 2.43

# %%
# %reload_ext autoreload
# %autoreload 2

import tiingo_fetcher
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import backtest
import ledoit_wolf
from datetime import date
from dotenv import load_dotenv
import monthly_regression
import os

# Make charts interactive
# # %matplotlib widget

# %%
load_dotenv()
api_key = os.getenv('TIINGO_API_KEY')

# 3× leveraged
# instruments = ['UPRO', 'UGL', 'TMF', 'TMV', 'AGQ', 'QLD', 'UCO', 'CHAU']
instruments = ['UPRO', 'UGL', 'TMF', 'TMV']
# 2× leveraged
# instruments = ['SSO', 'UGL', 'UBT', 'TBT', 'AGQ']
# 1× leveraged
# instruments = ['SPY', 'GLD', 'TLT', 'TBF']

etf_data = tiingo_fetcher.fetch_data(api_key, instruments, end=date.today()) 

# %%
# Limit data (debug / devlop)
data = etf_data.loc[:]

print('Latest data')
print(data.tail(5))

# Only drop data *after* selecting instruments (or we will drop too much) 
data = data.dropna()

relative_closes = data / data.iloc[0]
relative_closes.plot(figsize=(15, 6))

latest_closes = data.iloc[-50:]
latest_relative_closes = latest_closes / latest_closes.iloc[0]
latest_relative_closes.plot(figsize=(15, 6))


# %%
# Get regression slopes for multiple timeframes, convert them to -1/1 and calculate their mean
trend_timeframes = [40, 60, 80, 100]

# An array with one df per timeframe
slopes = []

for timeframe in trend_timeframes:
    current_tf = f'{timeframe}D'
    print(f'Calculating trend for timeframe {current_tf}')
    slope = data.rolling(current_tf).apply(monthly_regression.get_relative_slope)
    slope = slope.dropna()
    slopes.append(slope)

# Convert every entry to -1, 0 or 1
binary_slopes = [
    np.sign(slope)
    for slope in slopes
]

# We must make sure that all dfs within the array have the same dimensions and indexes
common_index = binary_slopes[0].index
for df in binary_slopes[1:]:
    common_index = common_index.intersection(df.index)
common_binary_slopes = [df.loc[common_index] for df in binary_slopes]

# Get mean over all timeframes
trend = sum(common_binary_slopes) / len(trend_timeframes)
go_long = trend > 0
print(f'Most recent trends are:\n{trend.iloc[-5:]}')

# Try an approach where we go long when our sum of trends is in the top 1/3 over the last 2 years
# trend_sum = sum(df.loc[common_index] for df in slopes)
# threshold = trend_sum.rolling('730D').quantile(0.8)
# go_long = trend_sum >= threshold
# print(f'Quantile based long positions are\n{go_long.tail(5)}')

# %%
# Just print the instruments
instruments = trend.columns
for instrument in instruments:
  instrument_data = pd.DataFrame({
    'trend': trend[instrument],
    'data': data[instrument]
  })
  # We only get trends for the first day of the month; every other day will be NaN. Use the previous
  # value for all subsequent days of the month.
  instrument_data['trend'] = instrument_data['trend'].ffill()
  fig, ax1 = plt.subplots(figsize=(16, 8))
  ax2 = ax1.twinx()
  ax2.plot(instrument_data.trend, color='#ADD8E6')
  ax1.plot(instrument_data.data)
  plt.title(instrument)
  plt.show()


# %%
weight_list = []
lookback_delta = pd.Timedelta(days=60)

# Only start after lookback_days have passed to make sure that the blocks are not too small at
# the beginning.
first_date = data.index[0] + lookback_delta

# Get the Ledoit Wolf weights; only use the instruments that have a positive trend, discard the
# rest *before* calculating the weights (Ledoit Wolf uses covariance; passing in unused instruments
# would invalidate the matrix to some extent)
for end_date in common_index[common_index > first_date]:
    start_date = end_date - lookback_delta
    # Only use last x days (every block gets longer)
    current_block = data[data.index > start_date].loc[:end_date]
    print(f'Calculating weights from {start_date} to {end_date} {len(current_block)} rows.')
    # Only pass thrending instruments to Ledoit Wolf
    current_long = go_long.loc[end_date]
    long_instruments = current_long[current_long].index.values
    print(f'Long on {end_date}: {long_instruments}')
    current_block_with_uptrend = current_block[long_instruments]
    if long_instruments.size == 0:
        continue
    weights = ledoit_wolf.calculate_weights(current_block_with_uptrend)
    series = pd.Series(weights, index=current_block_with_uptrend.columns, name=current_block.index[-1])
    weight_list.append(series)

weights = pd.concat(weight_list, axis=1).T
weights = weights.fillna(0)

# Test equal weights 
# weights[:] = 1 / len(data.columns)
# weights[trend < 0] = 0
# weights = weights.mul(1 / weights.sum(axis = 1), axis = 0)

weights.iloc[-52:].plot(kind='bar', stacked=True, figsize=(16, 8))
weights.iloc[-52:].sum(axis = 1).plot()

print(f'Latest weights:\n{weights.tail(5)}')

# %%
# Limit exposition to relative weight of instruments with an uptrend (if 3/4 instruments have
# an uptrend, expose 75%)
# Trends' mean flucutates between -1 (all negative) and 1 (all positive); adjust to go from 0 to 1
max_exposition = (trend.mean(axis=1) + 1) / 2
# max_exposition[:] = 1

# max_exposition is usually max. 0.75 because TMF and TMV are contrarian. Let's push it a bit so
# that we get 100% exposition when 3 of 4 instruments have an uptrend (if not, we limit ourselves
# to 0.75 most of the time)
max_exposition = max_exposition * 4 / 3
max_exposition[max_exposition > 1] = 1
# max_exposition[:] = 1

# When either TMV *OR* TMF are true, go to 100%
# consolidated_trend = go_long.astype(bool)
# consolidated_trend['TM'] = consolidated_trend['TMF'] | consolidated_trend['TMV']
# consolidated_trend = consolidated_trend.drop(columns=['TMF', 'TMV'])
# print(consolidated_trend)
# max_exposition = (consolidated_trend.mean(axis=1) + 1) / 2

# Limit max expsoition
adjusted_weights = weights.mul(max_exposition, axis=0)

print('Adjusted Weights:')
formatted = series.apply(lambda x: f"{x * 100:.2f}%")
print(adjusted_weights.iloc[-1].apply(lambda x: f'{x * 100:.1f}%').to_string()) 
print('---')

account_value = 200_000

latest_quote = data.iloc[-1]
print('Latest Quotes:')
print(latest_quote.to_string())
print('---')
distribution = account_value * adjusted_weights.iloc[-1]
print('Distribution:')
print(distribution.apply(np.floor).to_string())
print('---')
print('Positions:')
print((distribution / latest_quote).apply(np.floor).to_string())
print('---')

adjusted_weights.sum(axis=1).plot(figsize=(16, 3))
max_exposition.plot()
adjusted_weights.plot(kind='bar', stacked=True, figsize=(16, 8))

print(adjusted_weights.iloc[-5:])

# %%
# bt needs dates as datetimes
data.index = pd.to_datetime(data.index)
adjusted_weights.index = pd.to_datetime(adjusted_weights.index)

# Limit backtest
# data = data[:'2024-12-31']
# adjusted_weights = adjusted_weights[:'2024-12-31']

result = backtest.run('ledoit_wolf', data, adjusted_weights)

result.display()
result.plot(figsize=(16, 8))

# Let's display how we should have done in the past month (just to check reality vs. strategy)
first_of_month = result.prices.groupby(result.prices.index.to_period('M')).head(1)
first_of_month_and_previous = pd.DataFrame(first_of_month)
first_of_month_and_previous['PreviousMonth'] = first_of_month_and_previous.shift(1)
for index, row in first_of_month_and_previous.iterrows():
  print(f'{index.date()}, {(row['ledoit_wolf'] / row['PreviousMonth']) * 100 - 100:.2f}%') 

rolling_max = result.prices.cummax()
drawdowns = (result.prices - rolling_max) / rolling_max
drawdowns.plot(figsize=(16, 4))
plt.show()

recent_results = result.prices.loc[result.prices.index[-1] - pd.Timedelta(days=(60)):]
recent_results.plot(figsize=(16, 4), title='Recent results')
recent_rolling_max = recent_results.cummax()
recent_drawdowns = (recent_results - recent_rolling_max) / recent_rolling_max
recent_drawdowns.plot(figsize=(16, 4), title='Recent drawdowns')
plt.show()
