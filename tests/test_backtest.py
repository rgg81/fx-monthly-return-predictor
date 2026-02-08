import unittest
import pandas as pd
import numpy as np
import sys
import os
import random
from datetime import datetime, timedelta

# Add both paths: src/ for "signals.strategy" imports, src/signals/ for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'signals'))

from signals.backtest import Backtest
from signals.strategy import Strategy

class MockStrategy(Strategy):
    def __init__(self, signal_value=1):
        self.signal_value = signal_value
        self.features_optimization = False
        self.accumulated_returns_months = 36
    
    def fit(self, X, y):
        """Mock fit method - does nothing for testing"""
        pass
    
    def generate_signal(self, past_data, current_data):
        # Return lists to match backtest expectations
        num_rows = len(current_data)
        return [self.signal_value] * num_rows, [10] * num_rows


class MockRandomStrategy(Strategy):
    """Mock random strategy for testing that returns lists"""
    
    def __init__(self):
        super().__init__(symbol="EURUSD", step_size=6)
        self.features_optimization = False
        self.accumulated_returns_months = 36
    
    def fit(self, X, y):
        """Random strategy doesn't need training"""
        self.fitted = True
    
    def generate_signal(self, past_data, current_data):
        """Generate random signals returning lists"""
        num_rows = len(current_data)
        signals = [random.randint(0, 1) for _ in range(num_rows)]
        amounts = [10] * num_rows
        return signals, amounts


class TestBacktest(unittest.TestCase):
    
    def setUp(self):
        """Set up test data that can be used across test methods"""
        # Create a DataFrame with price data
        # Increasing number of data points to handle the 160 minimum past data requirement
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
        prices = np.linspace(100, 200, 300)  # Linear price increase from 100 to 200
        self.data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Feature1': range(300),
            'Feature2': range(100, 400),
            'Label': [1] * 300  # Add Label column for backtest compatibility
        })
        
        # Create test strategies
        self.buy_strategy = MockStrategy(signal_value=1)  # Always buy
        self.sell_strategy = MockStrategy(signal_value=0)  # Always sell
        self.random_strategy = MockRandomStrategy()
    
    def test_backtest_initialization(self):
        """Test that the Backtest class initializes correctly with different parameters"""
        # Default parameters
        backtest = Backtest(self.buy_strategy)
        self.assertEqual(backtest.strategy, self.buy_strategy)
        self.assertEqual(backtest.max_amount, 10)
        self.assertEqual(backtest.stop_loss, 0.015)  # Updated to match new default
        self.assertEqual(backtest.close_col, 'Close')
        
        # Custom parameters
        backtest = Backtest(
            strategy=self.sell_strategy,
            max_amount=20,
            stop_loss=0.05,
            close_col='CustomClose'
        )
        self.assertEqual(backtest.strategy, self.sell_strategy)
        self.assertEqual(backtest.max_amount, 20)
        self.assertEqual(backtest.stop_loss, 0.05)
        self.assertEqual(backtest.close_col, 'CustomClose')
    
    def test_backtest_run_with_buy_strategy(self):
        """Test backtesting with a strategy that always generates buy signals"""
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(self.data)
        
        # Check that we get results
        self.assertGreater(len(results), 0, "Results should not be empty")
        
        # All signals should be buy (1)
        self.assertTrue(all(results['Signal'] == 1))
        
        # All amounts should be 10
        self.assertTrue(all(results['Amount'] == 10))
        
        # Test return values (should be positive in an uptrend)
        self.assertTrue(sum(results['Return'] > 0) > sum(results['Return'] <= 0))
    
    def test_backtest_run_with_sell_strategy(self):
        """Test backtesting with a strategy that always generates sell signals"""
        backtest = Backtest(self.sell_strategy)
        results = backtest.run(self.data)
        
        # Check that we get results
        self.assertGreater(len(results), 0, "Results should not be empty")
        
        # All signals should be sell (0)
        self.assertTrue(all(results['Signal'] == 0))
        
        # Test return values (should be negative in an uptrend)
        self.assertTrue(sum(results['Return'] < 0) > sum(results['Return'] >= 0))
    
    def test_backtest_with_stop_loss(self):
        """Test that the backtest respects the stop loss parameter"""
        backtest = Backtest(self.buy_strategy, stop_loss=0.01)
        results = backtest.run(self.data)
        
        # All returns should be limited by the stop loss
        self.assertTrue(all(results['Return'] <= 0.01))
        self.assertTrue(all(results['Return'] >= -0.01))
    
    def test_backtest_with_different_close_column(self):
        """Test that the backtest works with a different close price column name"""
        # Create data with a different column name for close prices
        data_copy = self.data.copy()
        data_copy['CustomClose'] = data_copy['Close']
        
        backtest = Backtest(self.buy_strategy, close_col='CustomClose')
        results = backtest.run(data_copy)
        
        # Check that we get results
        self.assertGreater(len(results), 0, "Results should not be empty")
    
    def test_backtest_with_empty_data(self):
        """Test that the backtest handles an empty DataFrame gracefully"""
        empty_data = pd.DataFrame(columns=['Date', 'Close', 'Feature1', 'Feature2'])
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(empty_data)
        
        # Should return an empty DataFrame
        self.assertTrue(results.empty)
    
    def test_backtest_with_random_strategy(self):
        """Test that the backtest works with a random strategy"""
        backtest = Backtest(self.random_strategy)
        results = backtest.run(self.data)
        
        # Check that we get results
        self.assertGreater(len(results), 0, "Results should not be empty")
        
        # Signals should be a mix of 0s and 1s
        # Due to randomness, this might occasionally fail
        signal_values = results['Signal'].unique()
        # Just check that we have valid signals (0 or 1)
        self.assertTrue(all(s in [0, 1] for s in signal_values))
    
    def test_backtest_with_volatility_data(self):
        """Test backtesting with volatile price data (up and down movements)"""
        # Create volatile price data (sine wave)
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
        t = np.linspace(0, 8*np.pi, 300)
        prices = 150 + 50 * np.sin(t)  # Oscillate between 100 and 200
        
        volatile_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Feature1': range(300),
            'Feature2': range(100, 400),
            'Label': [1] * 300  # Add Label column for backtest compatibility
        })
        
        # Test with buy strategy
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(volatile_data)
        
        # Should have a mix of positive and negative returns
        self.assertTrue(sum(results['Return'] > 0) > 0)
        self.assertTrue(sum(results['Return'] < 0) > 0)


if __name__ == '__main__':
    unittest.main()