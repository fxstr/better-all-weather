import requests
import json
import pandas as pd
from datetime import date

def fetch_data(api_key, symbols, start='1990-01-01', end=date.today()):
    """
    Don't use pandas_datareader; it's broken and no-one fixes it:
    https://github.com/pydata/pandas-datareader/issues/965

    Fetches multiple symbols from Tiingo and returns them as a Pandas DataFrame, where every
    symbol's close is a column and every date a row

    Parameters
    ----------
    symbols: list of strings
        symbols to fetch
    start: string, default=1990-01-01
        start date
    end: string, default=date.today()
        end date

    Returns
    -------
    Pandas.DataFrame

    Raises
    ------
    date Exception
        If for at least one symbol there's no data available (thrown by pandas_datareader)
    """

    print(f'Get data for {symbols}, from {start} to {end}')
    base_url = "https://api.tiingo.com/tiingo/daily"

    series = []

    for symbol in symbols:
        url = f"{base_url}/{symbol}/prices"
    
        params = {
            "startDate": start,
            "endDate": end,
            "token": api_key
        }
        
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        current_series = pd.DataFrame(
            [
                (d['date'], d['adjClose']) for d in response.json()
            ],
            columns = ['date', symbol],
        )
        # Convert'2025-01-02T00:00:00.000Z' to a date (no time, no timezone)
        current_series['date'] = pd.to_datetime(current_series['date']).dt.tz_localize(None).dt.date
        current_series = current_series.set_index('date')
        series.append(current_series)

        print(f'Got data for {symbol} from {current_series.index[0]} to {current_series.index[-1]}')

    df = pd.concat(series, axis=1)
    df = df.sort_index()

    # Data for some symbols may not be available at the given start date; fill those values with
    # 0 (or bt will fail); this might cause trouble if in the middle of a series – we should
    # therefore maybe use ffill instead
    df = df.ffill()
    print(f'Returning a DataFrame of {df.shape[1]}×{df.shape[0]}')
    return df


# Simple test from within the file
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    etfs = ["SPY", "QQQ", "IWM"]
    result = fetch_data(os.getenv("TIINGO_API_KEY"), etfs, '2025-01-01', '2025-04-01')
    print(result)