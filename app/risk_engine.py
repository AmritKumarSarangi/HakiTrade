import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, portfolio_capital):
        """
        Institutional Risk Engine
        :param portfolio_capital: Total account equity in base currency
        """
        self.portfolio_capital = portfolio_capital
        
        # Risk Parameters defined by User
        self.max_position_size = 0.10     # Max 10% of portfolio in one asset
        self.max_daily_var = 0.015        # Max 1.5% portfolio Value at Risk (95%)
        self.max_open_positions = 10      # Max 10 concurrent trades
        self.default_stop_loss = 0.05     # 5% hard stop loss
        self.max_sector_exposure = 0.25   # Max 25% in one sector
        self.cash_reserve = 0.10          # Min 10% cash reserve
        
        # Simple sector mapping dictionary for the tracked stocks
        self.sector_map = {
            "RELIANCE.NS": "Energy",
            "TCS.NS": "IT",
            "INFY.NS": "IT",
            "HDFCBANK.NS": "Financials",
            "ICICIBANK.NS": "Financials"
        }

    def calculate_historical_var(self, current_allocations, new_trade_symbol, new_trade_weight, confidence_level=0.95):
        """
        Calculates the portfolio Value at Risk (VaR) using historical simulation 
        if the new trade is executed.
        """
        # Combine existing and new trade weights
        symbols = list(current_allocations.keys())
        weights = list(current_allocations.values())
        
        if new_trade_symbol in symbols:
            idx = symbols.index(new_trade_symbol)
            weights[idx] += new_trade_weight
        else:
            symbols.append(new_trade_symbol)
            weights.append(new_trade_weight)
            
        weights = np.array(weights)
        
        # Fetch 1 year of historical data
        try:
            data = yf.download(symbols, period="1y", interval="1d", progress=False)['Close']
            # Handle single symbol case vs multi symbol DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name=symbols[0])
                
            returns = data.pct_change().dropna()
            
            # Calculate historical daily portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Find the 5th percentile return (Historical VaR at 95% confidence)
            var_95 = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # VaR is typically expressed as a positive number representing loss
            return abs(var_95)
            
        except Exception as e:
            logger.error(f"Failed to calculate VaR: {e}")
            return None

    def evaluate_trade(self, symbol, intended_capital, current_portfolio):
        """
        Evaluates a proposed trade against strict risk rules.
        :param symbol: Ticker symbol
        :param intended_capital: Dollar/Rupee amount to invest
        :param current_portfolio: Dict with 'open_positions' (list of symbols), 
                                  'allocations' (dict of symbol: % weight), 
                                  'sector_exposure' (dict of sector: % weight)
        :return: Dict indicating if trade is approved and reason if rejected.
        """
        new_weight = intended_capital / self.portfolio_capital
        existing_weight = current_portfolio.get('allocations', {}).get(symbol, 0.0)
        total_proposed_weight = existing_weight + new_weight
        
        sector = self.sector_map.get(symbol, "Unknown")
        existing_sector_weight = current_portfolio.get('sector_exposure', {}).get(sector, 0.0)
        
        # 1. Open Positions Limit
        is_new_position = symbol not in current_portfolio.get('open_positions', [])
        if is_new_position and len(current_portfolio.get('open_positions', [])) >= self.max_open_positions:
            return {"approved": False, "reason": f"Max open positions ({self.max_open_positions}) reached."}
            
        # 2. Max Position Size Limit
        if total_proposed_weight > self.max_position_size:
            return {"approved": False, "reason": f"Trade pushes {symbol} weight ({total_proposed_weight:.1%}) above max limit ({self.max_position_size:.1%})."}
            
        # 3. Sector Exposure Limit
        if existing_sector_weight + new_weight > self.max_sector_exposure:
            return {"approved": False, "reason": f"Trade pushes {sector} sector weight above max limit ({self.max_sector_exposure:.1%})."}
            
        # 4. Cash Reserve Limit
        total_invested = sum(current_portfolio.get('allocations', {}).values())
        if total_invested + new_weight > (1.0 - self.cash_reserve):
            return {"approved": False, "reason": f"Trade violates minimum cash reserve requirement ({self.cash_reserve:.1%})."}
            
        # 5. Portfolio VaR Limit (Value at Risk)
        var = self.calculate_historical_var(current_portfolio.get('allocations', {}), symbol, new_weight)
        if var and var > self.max_daily_var:
            return {"approved": False, "reason": f"Trade pushes portfolio 95% VaR ({var:.2%}) above limit ({self.max_daily_var:.2%})."}
            
        return {
            "approved": True, 
            "reason": "Passed all risk checks.",
            "recommended_stop_loss_pct": self.default_stop_loss
        }

if __name__ == "__main__":
    # Quick Test
    rm = RiskManager(portfolio_capital=1000000) # ₹1,000,000 portfolio
    
    current_state = {
        "open_positions": ["TCS.NS", "HDFCBANK.NS"],
        "allocations": {"TCS.NS": 0.08, "HDFCBANK.NS": 0.05},
        "sector_exposure": {"IT": 0.08, "Financials": 0.05}
    }
    
    # Try buying ₹80,000 of INFY (8% of portfolio)
    # This should fail because IT sector would jump to 8% + 8% = 16%, 
    # but wait, sector limit is 25%, so it passes sector check.
    # What if we try to buy 15% of INFY? Fails position limit (10%).
    
    print("\n--- Testing Risk Engine ---")
    print("Test 1: Buying 15% of INFY (Should fail position limit)")
    print(rm.evaluate_trade("INFY.NS", 150000, current_state))
    
    print("\nTest 2: Buying 8% of INFY (Should pass limits)")
    print(rm.evaluate_trade("INFY.NS", 80000, current_state))
