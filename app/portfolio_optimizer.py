import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self, symbols, max_position_size=0.10, cash_reserve=0.10, risk_free_rate=0.06):
        """
        Markowitz Mean-Variance Portfolio Optimizer
        :param symbols: List of ticker symbols
        :param max_position_size: Maximum % of total portfolio allowed in a single asset
        :param cash_reserve: Minimum % of portfolio kept in cash
        :param risk_free_rate: Annualized risk-free rate (e.g., 6% for Indian Bonds)
        """
        self.symbols = symbols
        self.max_position_size = max_position_size
        self.cash_reserve = cash_reserve
        self.risk_free_rate = risk_free_rate
        
        # Calculate maximum possible investment based on universe size
        max_possible_investment = len(symbols) * max_position_size
        self.target_investment = min(1.0 - cash_reserve, max_possible_investment)
        
        if self.target_investment < (1.0 - cash_reserve):
            logger.warning(f"Universe too small ({len(symbols)} stocks) to deploy {1-cash_reserve:.0%} capital given the {max_position_size:.0%} position limit. Max deployment will be {self.target_investment:.0%}.")

    def fetch_historical_data(self, period="1y"):
        """Fetches adjusted close prices for all symbols to calculate covariance."""
        data = yf.download(self.symbols, period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            prices = data
        return prices.dropna(how='all')

    def optimize(self):
        """Calculates optimal weights to maximize the Sharpe Ratio."""
        prices = self.fetch_historical_data()
        
        # Calculate daily logarithmic returns
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Annualized expected returns and covariance matrix (252 trading days)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        num_assets = len(self.symbols)
        
        def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            # Calculate portfolio return and volatility
            p_ret = np.sum(mean_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Maximize Sharpe = Minimize Negative Sharpe
            return -(p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0
            
        # Initial guess (equal distribution of the target investment)
        initial_weights = np.array([self.target_investment / num_assets] * num_assets)
        
        # Bounds: 0 to max_position_size for each asset
        bounds = tuple((0.0, self.max_position_size) for _ in range(num_assets))
        
        # Constraint: Sum of weights must equal the target investment
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - self.target_investment})
        
        # Run optimization
        result = minimize(
            negative_sharpe, 
            initial_weights, 
            args=(mean_returns.values, cov_matrix.values, self.risk_free_rate),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            return None
            
        optimal_weights = result.x
        
        # Compile results
        allocation = {self.symbols[i]: round(optimal_weights[i], 4) for i in range(num_assets)}
        allocation["CASH"] = round(1.0 - self.target_investment, 4)
        
        expected_return = np.sum(mean_returns * optimal_weights)
        expected_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        optimal_sharpe = (expected_return - self.risk_free_rate) / expected_volatility
        
        return {
            "allocation": allocation,
            "metrics": {
                "expected_annual_return": round(expected_return * 100, 2),
                "annual_volatility": round(expected_volatility * 100, 2),
                "sharpe_ratio": round(optimal_sharpe, 2)
            }
        }

if __name__ == "__main__":
    # Quick Test
    STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    optimizer = PortfolioOptimizer(STOCKS)
    result = optimizer.optimize()
    print("\n--- Optimized Portfolio ---")
    for k, v in result['allocation'].items():
        print(f"{k}: {v:.1%}")
    print("\nMetrics:", result['metrics'])
