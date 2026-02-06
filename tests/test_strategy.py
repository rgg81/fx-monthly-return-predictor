import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'signals'))
from strategy import Strategy, RandomStrategy


class ConcreteStrategy(Strategy):
    """Concrete implementation for testing abstract Strategy class"""
    
    def __init__(self, symbol="EURUSD", step_size=6, feature_set=None, feature_frequency=None):
        super().__init__(symbol=symbol, step_size=step_size, feature_set=feature_set, feature_frequency=feature_frequency)
        self.model = None
    
    def fit(self, X, y):
        self.fitted = True
        self.model = "trained"
    
    def generate_signal(self, past_data, current_data):
        return 1, 10


class TestStrategyBase(unittest.TestCase):
    """Test cases for Strategy base class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = ConcreteStrategy(symbol="EURUSD", step_size=6)
        
        dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'EURUSD_Close': np.random.randn(100) + 1.1,
            'Label': np.random.randint(0, 2, 100),
            'tech_rsi_14': np.random.randn(100),
            'tech_macd_12': np.random.randn(100),
            'mr_bb_width_6': np.random.randn(100),
            'mr_momentum_18': np.random.randn(100),
            'regime_volatility_6': np.random.randn(100),
            'regime_trend_12': np.random.randn(100),
            'macro_spread_18': np.random.randn(100)
        })
    
    def test_initialization(self):
        """Test strategy initialization with various parameters"""
        strategy = ConcreteStrategy(symbol="XAUUSD", step_size=12, feature_set="tech_", feature_frequency="_14")
        
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.step_size, 12)
        self.assertEqual(strategy.feature_set, "tech_")
        self.assertEqual(strategy.feature_frequency, "_14")
        self.assertFalse(strategy.fitted)
    
    def test_default_initialization(self):
        """Test strategy with default parameters"""
        strategy = ConcreteStrategy()
        
        self.assertEqual(strategy.symbol, "EURUSD")
        self.assertEqual(strategy.step_size, 6)
        self.assertIsNone(strategy.feature_set)
        self.assertIsNone(strategy.feature_frequency)
    
    def test_filter_features_no_filter(self):
        """Test feature filtering with no filters applied"""
        filtered_data = self.strategy._filter_features(self.test_data)
        
        self.assertEqual(len(filtered_data.columns), len(self.test_data.columns))
        self.assertIn('Label', filtered_data.columns)
        self.assertIn('Date', filtered_data.columns)
        self.assertIn('EURUSD_Close', filtered_data.columns)
    
    def test_filter_features_by_prefix(self):
        """Test feature filtering by feature_set prefix"""
        strategy = ConcreteStrategy(symbol="EURUSD", feature_set="tech_")
        filtered_data = strategy._filter_features(self.test_data)
        
        self.assertIn('Label', filtered_data.columns)
        self.assertIn('Date', filtered_data.columns)
        self.assertIn('EURUSD_Close', filtered_data.columns)
        self.assertIn('tech_rsi_14', filtered_data.columns)
        self.assertIn('tech_macd_12', filtered_data.columns)
        self.assertNotIn('mr_bb_width_6', filtered_data.columns)
        self.assertNotIn('regime_volatility_6', filtered_data.columns)
    
    def test_filter_features_by_frequency(self):
        """Test feature filtering by frequency suffix"""
        strategy = ConcreteStrategy(symbol="EURUSD", feature_frequency="_6")
        filtered_data = strategy._filter_features(self.test_data)
        
        self.assertIn('Label', filtered_data.columns)
        self.assertIn('Date', filtered_data.columns)
        self.assertIn('EURUSD_Close', filtered_data.columns)
        self.assertIn('mr_bb_width_6', filtered_data.columns)
        self.assertIn('regime_volatility_6', filtered_data.columns)
        self.assertNotIn('tech_rsi_14', filtered_data.columns)
        self.assertNotIn('mr_momentum_18', filtered_data.columns)
    
    def test_filter_features_by_prefix_and_frequency(self):
        """Test feature filtering by both prefix and frequency"""
        strategy = ConcreteStrategy(symbol="EURUSD", feature_set="regime_", feature_frequency="_6")
        filtered_data = strategy._filter_features(self.test_data)
        
        self.assertIn('Label', filtered_data.columns)
        self.assertIn('Date', filtered_data.columns)
        self.assertIn('EURUSD_Close', filtered_data.columns)
        self.assertIn('regime_volatility_6', filtered_data.columns)
        self.assertNotIn('regime_trend_12', filtered_data.columns)
        self.assertNotIn('tech_rsi_14', filtered_data.columns)
        self.assertNotIn('mr_bb_width_6', filtered_data.columns)
    
    def test_fit_method(self):
        """Test fit method sets fitted flag"""
        X = self.test_data.drop(['Label', 'Date'], axis=1)
        y = self.test_data['Label']
        
        self.assertFalse(self.strategy.fitted)
        self.strategy.fit(X, y)
        self.assertTrue(self.strategy.fitted)
    
    def test_generate_signal(self):
        """Test generate_signal returns correct format"""
        past_data = self.test_data.iloc[:50]
        current_data = self.test_data.iloc[50:51]
        
        signal, amount = self.strategy.generate_signal(past_data, current_data)
        
        self.assertIn(signal, [0, 1])
        self.assertIsInstance(amount, int)
        self.assertGreater(amount, 0)


class TestRandomStrategy(unittest.TestCase):
    """Test cases for RandomStrategy implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = RandomStrategy(symbol="GBPUSD", step_size=3)
    
    def test_random_strategy_initialization(self):
        """Test RandomStrategy initialization"""
        self.assertEqual(self.strategy.symbol, "GBPUSD")
        self.assertEqual(self.strategy.step_size, 3)
        self.assertIsNone(self.strategy.feature_set)
        self.assertIsNone(self.strategy.feature_frequency)
    
    def test_random_strategy_fit(self):
        """Test RandomStrategy fit method"""
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        self.assertFalse(self.strategy.fitted)
        self.strategy.fit(X, y)
        self.assertTrue(self.strategy.fitted)
    
    def test_random_strategy_generate_signal(self):
        """Test RandomStrategy generates valid signals"""
        past_data = pd.DataFrame({'dummy': [1, 2, 3]})
        current_data = pd.DataFrame({'dummy': [4]})
        
        signal, amount = self.strategy.generate_signal(past_data, current_data)
        
        self.assertIn(signal, [0, 1])
        self.assertEqual(amount, 10)
    
    def test_random_strategy_signal_randomness(self):
        """Test RandomStrategy produces varying signals"""
        past_data = pd.DataFrame({'dummy': [1, 2, 3]})
        current_data = pd.DataFrame({'dummy': [4]})
        
        signals = []
        for _ in range(100):
            signal, amount = self.strategy.generate_signal(past_data, current_data)
            signals.append(signal)
        
        self.assertTrue(0 in signals and 1 in signals, "Should generate both buy and sell signals")
        self.assertTrue(20 < sum(signals) < 80, "Should be roughly balanced between 0 and 1")


class TestStrategyWithDifferentSymbols(unittest.TestCase):
    """Test strategy behavior with different trading symbols"""
    
    def test_eurusd_symbol(self):
        """Test with EURUSD symbol"""
        strategy = ConcreteStrategy(symbol="EURUSD")
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='M'),
            'EURUSD_Close': np.random.randn(10),
            'Label': np.random.randint(0, 2, 10)
        })
        
        filtered = strategy._filter_features(data)
        self.assertIn('EURUSD_Close', filtered.columns)
    
    def test_xauusd_symbol(self):
        """Test with XAUUSD symbol"""
        strategy = ConcreteStrategy(symbol="XAUUSD")
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='M'),
            'XAUUSD_Close': np.random.randn(10),
            'Label': np.random.randint(0, 2, 10)
        })
        
        filtered = strategy._filter_features(data)
        self.assertIn('XAUUSD_Close', filtered.columns)


class TestAbstractMethods(unittest.TestCase):
    """Test that abstract methods cannot be instantiated"""
    
    def test_cannot_instantiate_base_strategy(self):
        """Test that Strategy base class cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            strategy = Strategy()


if __name__ == '__main__':
    unittest.main(verbosity=2)
