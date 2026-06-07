import marimo

__generated_with = "0.23.9"
app = marimo.App(width="wide")


# Imports: all third-party and local modules
@app.cell
def _():
    # Variables prefixed with _ are cell-private in Marimo: they are not exported into
    # the reactive graph and cannot be referenced by other cells. Only names listed in
    # `return` are shared. Use _ for anything that is intermediate or display-only.
    import os
    from datetime import date

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    import backtest
    import ledoit_wolf
    import monthly_regression
    import plot_utils
    import tiingo_fetcher

    return (
        backtest,
        date,
        ledoit_wolf,
        load_dotenv,
        mo,
        monthly_regression,
        np,
        os,
        pd,
        plot_utils,
        plt,
        tiingo_fetcher,
    )


# Introduction: strategy description and performance expectations
@app.cell
def _(mo):
    mo.md(r"""
    # Ledoit-Wolf Trend Strategy

    - Get data from Tiingo
    - Use linear regression slope of multiple timeframes to determine every instrument's trend.
      Across every timeframe, we use -1 for a negative trend and +1 for a positive one to account for
      the steeper slopes of shorter timeframes. We then use the mean across those -1/1 values to
      determine the trend.
    - Pass all instruments with positive trends to the weight allocation step.
      We use equal weight across trending instruments.
    - We average the trends; this results in a number between -1 and 1; we adjust it to go from 0 to 1
      which gives us the target exposure; we then multiply our weights by it to get the adjusted weights.

    We could go weekly with trend [20, 30, 40, 50, 60] and 30 lookback for Ledoit-Wolf.

    ## Expectations
    - Triple ETFs & max. exp. = long / all: 12% CAGR, 29% MaxDD → 2.42
    - Triple ETFs & max. exp. = 100%: 16% CAGR, 39% MaxDD → 2.43
    - Double ETFs & max. exp. = 100%: 13% CAGR, 38% MaxDD → 2.43
    """)
    return


# Config: API key and instrument selection
@app.cell
def _(load_dotenv, os):
    load_dotenv()
    api_key = os.getenv("TIINGO_API_KEY")
    # 3× leveraged
    instruments = ["UPRO", "UGL", "TMF", "TMV"]
    # 2× leveraged
    # instruments = ['SSO', 'UGL', 'UBT', 'TBT', 'AGQ']
    # 1× leveraged
    # instruments = ['SPY', 'GLD', 'TLT', 'TBF']
    return api_key, instruments


# Data: fetch adjusted close prices from Tiingo
@app.cell
def _(api_key, date, instruments, tiingo_fetcher):
    _raw = tiingo_fetcher.fetch_data(api_key, instruments, end=date.today())
    # Drop rows where any instrument has no data yet (e.g. TMV launched after UPRO)
    data = _raw.dropna()
    return (data,)


# Price charts: relative performance across full history and last 50 days
@app.cell
def _(data, mo, plot_utils, plt):
    # Normalize to the first row so all series start at 1.0,
    # making relative performance visually comparable across instruments
    _fig, _ax = plt.subplots(figsize=(15, 6))
    (data / data.iloc[0]).plot(ax=_ax, title="Relative closes (full history)")
    _html_full = plot_utils.fig_html(_fig)

    _last_50 = data.iloc[-50:]
    _fig, _ax = plt.subplots(figsize=(15, 6))
    (_last_50 / _last_50.iloc[0]).plot(ax=_ax, title="Relative closes (last 50 days)")
    _html_recent = plot_utils.fig_html(_fig)

    mo.vstack(
        [
            mo.md(f"**Latest data:**\n\n```\n{data.tail(5).to_string()}\n```"),
            _html_full,
            _html_recent,
        ]
    )
    return


# Trend: multi-timeframe regression slope → ±1 signal → mean trend per instrument
@app.cell
def _(data, monthly_regression, np):
    # Compute rolling regression slope for each timeframe and convert to ±1 sign.
    # Sign normalization is essential: raw slopes are not comparable across timeframes
    # because shorter windows produce steeper absolute slopes. ±1 puts all timeframes
    # on the same scale before averaging.
    _timeframes = [40, 60, 80, 100]
    _raw_slopes = [
        data.rolling(f"{tf}D").apply(monthly_regression.get_relative_slope).dropna()
        for tf in _timeframes
    ]
    _signed_slopes = [np.sign(s) for s in _raw_slopes]

    # Each timeframe may start on a slightly different date; intersect to get a common index
    common_index = _signed_slopes[0].index
    for _slope_df in _signed_slopes[1:]:
        common_index = common_index.intersection(_slope_df.index)
    _aligned_slopes = [s.loc[common_index] for s in _signed_slopes]

    # Mean across timeframes: -1 = all bearish, +1 = all bullish, 0 = mixed
    trend = sum(_aligned_slopes) / len(_timeframes)
    go_long = trend > 0
    return common_index, go_long, trend


# Trend charts: price overlaid with trend signal and holding periods (last 2 years)
@app.cell
def _(data, go_long, mo, pd, plot_utils, plt, trend):
    # Overlay trend signal on price for each instrument.
    # Trend is computed at month-end; ffill carries the signal forward through the month.
    _cutoff = data.index[-1] - pd.DateOffset(years=2)
    _trend_figs = []
    for _instrument in trend.columns:
        _price = data[_instrument][data.index >= _cutoff]
        _trend_signal = trend[_instrument].reindex(_price.index, method="ffill")
        _holding = (
            go_long[_instrument].reindex(_price.index, method="ffill").fillna(False)
        )

        _fig, _price_ax = plt.subplots(figsize=(16, 8))
        _trend_ax = _price_ax.twinx()
        _price_ax.plot(_price, label=_instrument)
        _trend_ax.plot(_trend_signal, color="#ADD8E6", label="trend")
        _price_ax.fill_between(
            _price.index,
            0,
            1,
            where=_holding,
            transform=_price_ax.get_xaxis_transform(),
            alpha=0.15,
            color="green",
        )
        _lines = _price_ax.get_legend_handles_labels()
        _lines_trend = _trend_ax.get_legend_handles_labels()
        _price_ax.legend(
            *[a + b for a, b in zip(_lines, _lines_trend)], loc="upper left"
        )
        _fig.suptitle(_instrument)
        _trend_figs.append(plot_utils.fig_html(_fig))

    mo.vstack(_trend_figs)
    return


# Weights: Ledoit-Wolf allocation across trending instruments at each rebalance date
@app.cell
def _(common_index, data, go_long, ledoit_wolf, pd):
    # For each rebalance date, pass ONLY trending instruments to the weight function.
    # Non-trending instruments receive zero weight.
    _lookback = pd.Timedelta(days=60)
    _first_eligible_date = data.index[0] + _lookback

    _weight_rows = []
    for _date in common_index[common_index > _first_eligible_date]:
        _lookback_window = data[data.index > _date - _lookback].loc[:_date]
        _is_trending_up = go_long.loc[_date]
        _trending_instruments = _is_trending_up[_is_trending_up].index.values
        if _trending_instruments.size == 0:
            continue
        _instrument_weights = ledoit_wolf.calculate_weights(
            _lookback_window[_trending_instruments]
        )
        _weight_rows.append(
            pd.Series(
                _instrument_weights,
                index=_trending_instruments,
                name=_lookback_window.index[-1],
            )
        )

    weights = pd.concat(_weight_rows, axis=1).T.fillna(0)
    return (weights,)


# Exposition: scale weights by trend strength and display allocation charts
@app.cell
def _(mo, plot_utils, plt, trend, weights):
    # Map mean trend [-1, 1] → exposition [0, 1], then scale by 4/3.
    # Without the scaling, 3 out of 4 instruments trending up only reaches 75% exposition
    # because TMF and TMV are counter-trend to each other, so one is always dragging the mean down.
    # The 4/3 factor pushes 3/4 bullish → 100% exposed.
    # reindex to weights.index: trend has a row for every month, but weights skips months
    # where no instrument was trending — reindex aligns them before multiplying.
    max_exposition = ((trend.reindex(weights.index).mean(axis=1) + 1) / 2 * (4 / 3)).clip(upper=1)
    adjusted_weights = weights.mul(max_exposition, axis=0)

    _w52 = weights.iloc[-52:]
    _fig, _ax = plt.subplots(figsize=(16, 8))
    _w52.plot(kind="bar", stacked=True, ax=_ax, title="Weights (last 52 periods)")
    plot_utils.set_date_ticks(_ax, _w52.index)
    _html_weights = plot_utils.fig_html(_fig)

    _fig, _ax = plt.subplots(figsize=(16, 3))
    _w52.sum(axis=1).plot(ax=_ax, title="Weight sum (last 52 periods)")
    _html_weight_sum = plot_utils.fig_html(_fig)

    _fig, _ax = plt.subplots(figsize=(16, 3))
    adjusted_weights.sum(axis=1).plot(ax=_ax, label="Adjusted exposure")
    max_exposition.plot(ax=_ax, label="Max exposition")
    _ax.legend()
    _ax.set_title("Exposition")
    _html_exposition = plot_utils.fig_html(_fig)

    _fig, _ax = plt.subplots(figsize=(16, 8))
    adjusted_weights.plot(kind="bar", stacked=True, ax=_ax, title="Adjusted weights")
    plot_utils.set_date_ticks(_ax, adjusted_weights.index)
    _html_adjusted = plot_utils.fig_html(_fig)

    mo.vstack([_html_weights, _html_weight_sum, _html_exposition, _html_adjusted])
    return adjusted_weights, max_exposition


# Account slider: interactive input for portfolio size
@app.cell
def _(mo):
    # Exported as a tuple so other cells can reference account_slider.value.
    # Must NOT be rendered again in any other cell (double render → "Invalid server token").
    account_slider = mo.ui.slider(
        start=50_000,
        stop=2_000_000,
        step=10_000,
        value=250_000,
        label="Account value ($)",
        show_value=True,
    )
    account_slider  # last expression = cell output; tuple return below exports the name
    return (account_slider,)


# Positions: current weights, prices, dollar allocation, and share counts
@app.cell
def _(account_slider, adjusted_weights, data, mo, np):
    _account_value = account_slider.value
    _latest_prices = data.iloc[-1]
    _dollar_allocation = _account_value * adjusted_weights.iloc[-1]
    _share_counts = (_dollar_allocation / _latest_prices).apply(np.floor)

    mo.vstack(
        [
            mo.md("## Current Positions"),
            mo.md(
                f"**Adjusted Weights:**\n```\n{adjusted_weights.iloc[-1].apply(lambda x: f'{x * 100:.1f}%').to_string()}\n```"
            ),
            mo.md(f"**Latest Prices:**\n```\n{_latest_prices.to_string()}\n```"),
            mo.md(
                f"**Dollar Allocation (account: ${_account_value:,}):**\n```\n{_dollar_allocation.apply(np.floor).to_string()}\n```"
            ),
            mo.md(f"**Share Counts:**\n```\n{_share_counts.to_string()}\n```"),
        ]
    )
    return


# Backtest: run bt simulation with adjusted weights
@app.cell
def _(adjusted_weights, backtest, data, pd):
    # bt requires DatetimeIndex; convert in private copies to avoid mutating shared state
    _prices = data.copy()
    _prices.index = pd.to_datetime(_prices.index)
    _weights = adjusted_weights.copy()
    _weights.index = pd.to_datetime(_weights.index)
    result = backtest.run("ledoit_wolf", _prices, _weights)
    return (result,)


# Results: performance, drawdowns, recent returns, and monthly bar chart
@app.cell
def _(mo, pd, plot_utils, plt, result):
    _fig, _ax = plt.subplots(figsize=(16, 8))
    result.prices.plot(ax=_ax, title="Strategy performance")
    _html_performance = plot_utils.fig_html(_fig)

    _fig, _ax = plt.subplots(figsize=(16, 4))
    plot_utils.drawdown(result.prices).plot(ax=_ax, title="Drawdowns")
    _html_drawdown = plot_utils.fig_html(_fig)

    _recent_prices = result.prices.loc[
        result.prices.index[-1] - pd.Timedelta(days=60) :
    ]
    _fig, _ax = plt.subplots(figsize=(16, 4))
    _recent_prices.plot(ax=_ax, title="Recent results (last 60 days)")
    _html_recent = plot_utils.fig_html(_fig)

    _fig, _ax = plt.subplots(figsize=(16, 4))
    plot_utils.drawdown(_recent_prices).plot(ax=_ax, title="Recent drawdowns")
    _html_recent_drawdown = plot_utils.fig_html(_fig)

    # Monthly return = first price of the month vs. first price of the previous month
    _month_open_prices = pd.DataFrame(
        result.prices.groupby(result.prices.index.to_period("M")).head(1)
    )
    _monthly_returns_pct = (
        _month_open_prices["ledoit_wolf"] / _month_open_prices["ledoit_wolf"].shift(1)
        - 1
    ).dropna() * 100
    _monthly_returns_pct = _monthly_returns_pct.iloc[-24:]
    _bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in _monthly_returns_pct]

    _fig, _ax = plt.subplots(figsize=(16, 4))
    _monthly_returns_pct.plot(
        kind="bar",
        ax=_ax,
        color=_bar_colors,
        title="Monthly Returns (%, last 24 months)",
    )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xlabel("")
    _ax.set_xticklabels(
        [p.strftime("%Y-%m") for p in _monthly_returns_pct.index],
        rotation=45,
        ha="right",
    )
    _html_monthly = plot_utils.fig_html(_fig)

    _stats_keys = [
        "start",
        "end",
        "total_return",
        "cagr",
        "max_drawdown",
        "calmar",
    ]
    mo.vstack(
        [
            mo.md(
                f"## Backtest Results\n\n**Stats:**\n```\n{result.stats.loc[_stats_keys].to_string()}\n```"
            ),
            _html_performance,
            _html_drawdown,
            _html_recent,
            _html_recent_drawdown,
            _html_monthly,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
