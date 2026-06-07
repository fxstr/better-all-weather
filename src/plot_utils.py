import marimo as mo
import matplotlib.pyplot as plt


# Convert a matplotlib figure to Marimo HTML and close it to free memory
def fig_html(fig):
    html = mo.as_html(fig)
    plt.close(fig)
    return html


# Set ~n evenly spaced date-only tick labels on a bar chart axis
def set_date_ticks(ax, index, n=12):
    ticks = range(0, len(index), max(1, len(index) // n))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(index[i].date()) for i in ticks], rotation=45, ha="right")


# Compute drawdown series (0 to -1) relative to the running peak
def drawdown(prices):
    peak = prices.cummax()
    return (prices - peak) / peak
