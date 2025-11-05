import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# Use Ledoit Wolf to get a Maximum Diversion portfolio
def calculate_weights(data):

    leverage = 1

    returns = data.pct_change(fill_method=None).dropna()

    # 1. Estimate the Ledoit-Wolf covariance matrix
    lw = LedoitWolf()
    cov_matrix = lw.fit(returns).covariance_   
    std_devs = np.sqrt(np.diag(cov_matrix))
    n = len(std_devs)

    # Objective: negative diversion ratio (to maximize it)
    def objective(weights):
        weighted_std = np.dot(weights, std_devs)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        diversion_ratio = weighted_std / portfolio_std
        return -diversion_ratio

    # Constraints: sum of weights == 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - leverage}

    # Bounds: no short selling (0 ≤ w ≤ 1)
    bounds = [(0, leverage) for _ in range(n)]

    # Initial guess: equal weight
    x0 = np.ones(n) / n

    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)