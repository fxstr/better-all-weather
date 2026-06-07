import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# Use Ledoit-Wolf covariance + Maximum Diversification objective
def calculate_weights(data):
    returns = data.pct_change(fill_method=None).dropna()

    cov = LedoitWolf().fit(returns).covariance_
    std_devs = np.sqrt(np.diag(cov))
    n = len(std_devs)

    # Objective: maximise diversification ratio (weighted avg vol / portfolio vol)
    def objective(weights):
        weighted_std = np.dot(weights, std_devs)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -(weighted_std / portfolio_std)

    result = minimize(
        objective,
        np.ones(n) / n,
        bounds=[(0, 1)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)