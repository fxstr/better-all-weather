import bt
import matplotlib.pyplot as plt

def run(strategy_name, data, weights):
  strategy = bt.Strategy(
      strategy_name,
      [
          # bt.algos.RunDaily(),
          # bt.algos.RunWeekly(),
          bt.algos.RunMonthly(),
          bt.algos.SelectAll(),
          bt.algos.WeighTarget(weights),
          bt.algos.Rebalance(),
      ],
  )

  test = bt.Backtest(strategy, data, initial_capital=100000)

  result = bt.run(
      test,
  )
  return result
