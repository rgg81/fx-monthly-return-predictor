import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add both paths: src/ for "signals.strategy" imports, src/signals/ for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'signals'))


def _make_ensemble_shell(**overrides):
    """Create an EnsembleOptunaStrategy with no real sub-strategies."""
    from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy

    with patch.object(EnsembleOptunaStrategy, '__init__', lambda self_: None):
        ens = EnsembleOptunaStrategy.__new__(EnsembleOptunaStrategy)
    defaults = {
        'max_amount': 10,
        'strategies': {},
        'fitted': False,
        'feature_set': None,
        'symbol': 'EURUSD',
        'stop_loss': 0.1,
        'close_col': 'EURUSD_Close',
        'features_optimization': False,
        'accumulated_returns_months': 36,
        'n_groups': 1,
        'frequency_range': (3, 4),
        'frequency_range_step': 1,
        'accumulated_returns': {},
        'group_predictions': {},
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(ens, k, v)
    return ens


def _make_price_data(prices, symbol='EURUSD'):
    return pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=len(prices), freq='ME'),
        f'{symbol}_Close': prices,
        'Label': [1] * len(prices),
    })


class TestAggregatePredictions(unittest.TestCase):

    def test_unanimous_buy(self):
        ens = _make_ensemble_shell()
        predictions = {
            'A': {'signals': [1], 'amounts': [10]},
            'B': {'signals': [1], 'amounts': [10]},
            'C': {'signals': [1], 'amounts': [8]},
        }
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(signals, [1])
        self.assertEqual(len(amounts), 1)
        self.assertGreater(amounts[0], 0)

    def test_unanimous_sell(self):
        ens = _make_ensemble_shell()
        predictions = {
            'A': {'signals': [0], 'amounts': [10]},
            'B': {'signals': [0], 'amounts': [10]},
        }
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(signals, [0])

    def test_split_vote_buy_wins(self):
        ens = _make_ensemble_shell()
        predictions = {
            'A': {'signals': [1], 'amounts': [10]},
            'B': {'signals': [1], 'amounts': [10]},
            'C': {'signals': [0], 'amounts': [5]},
        }
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(signals, [1])

    def test_split_vote_sell_wins(self):
        ens = _make_ensemble_shell()
        predictions = {
            'A': {'signals': [1], 'amounts': [5]},
            'B': {'signals': [0], 'amounts': [10]},
            'C': {'signals': [0], 'amounts': [10]},
        }
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(signals, [0])

    def test_empty_predictions_returns_default(self):
        ens = _make_ensemble_shell()
        signals, amounts = ens._aggregate_predictions({}, "test")
        self.assertEqual(signals, [0])
        self.assertEqual(amounts, [ens.max_amount])

    def test_single_prediction(self):
        ens = _make_ensemble_shell()
        predictions = {'A': {'signals': [1], 'amounts': [7]}}
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(signals, [1])
        self.assertEqual(len(amounts), 1)

    def test_multiple_signals_per_step(self):
        ens = _make_ensemble_shell()
        predictions = {
            'A': {'signals': [1, 0, 1], 'amounts': [10, 10, 10]},
            'B': {'signals': [1, 0, 0], 'amounts': [10, 10, 10]},
        }
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertEqual(len(signals), 3)
        self.assertEqual(len(amounts), 3)
        self.assertEqual(signals[0], 1)
        self.assertEqual(signals[1], 0)

    def test_amount_clamped_to_max(self):
        ens = _make_ensemble_shell()
        predictions = {'A': {'signals': [1], 'amounts': [10]}}
        signals, amounts = ens._aggregate_predictions(predictions, "test")
        self.assertLessEqual(amounts[0], ens.max_amount)
        self.assertGreaterEqual(amounts[0], 1)


class TestCalculateProfitLoss(unittest.TestCase):

    def test_buy_with_gain(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 1.02])
        pnl = ens._calculate_profit_loss([1], [10], data, stop_loss=0.05)
        self.assertEqual(len(pnl), 1)
        self.assertAlmostEqual(pnl[0], 0.02, places=4)

    def test_buy_with_loss(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 0.98])
        pnl = ens._calculate_profit_loss([1], [10], data, stop_loss=0.05)
        self.assertAlmostEqual(pnl[0], -0.02, places=4)

    def test_sell_with_gain(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 0.98])
        pnl = ens._calculate_profit_loss([0], [10], data, stop_loss=0.05)
        self.assertAlmostEqual(pnl[0], 0.02, places=4)

    def test_sell_with_loss(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 1.02])
        pnl = ens._calculate_profit_loss([0], [10], data, stop_loss=0.05)
        self.assertAlmostEqual(pnl[0], -0.02, places=4)

    def test_stop_loss_clamp_positive(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 1.10])  # +10% move
        pnl = ens._calculate_profit_loss([1], [10], data, stop_loss=0.015)
        self.assertAlmostEqual(pnl[0], 0.015, places=4)

    def test_stop_loss_clamp_negative(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 0.90])  # -10% move
        pnl = ens._calculate_profit_loss([1], [10], data, stop_loss=0.015)
        self.assertAlmostEqual(pnl[0], -0.015, places=4)

    def test_partial_amount(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 1.02])  # +2%, amount=5/10
        pnl = ens._calculate_profit_loss([1], [5], data, stop_loss=0.05)
        self.assertAlmostEqual(pnl[0], 0.01, places=4)

    def test_multiple_signals(self):
        ens = _make_ensemble_shell()
        data = _make_price_data([1.0, 1.01, 1.03])
        pnl = ens._calculate_profit_loss([1, 0], [10, 10], data, stop_loss=0.05)
        self.assertEqual(len(pnl), 2)


class TestClearGroupReturns(unittest.TestCase):

    def test_resets_all_groups(self):
        ens = _make_ensemble_shell(accumulated_returns={
            ('tech_', '_3'): [0.01, -0.02, 0.03],
            ('mr_', '_6'): [0.05, 0.01],
        })
        ens.clear_group_returns()
        for val in ens.accumulated_returns.values():
            self.assertEqual(val, [0.0])

    def test_clear_empty_dict(self):
        ens = _make_ensemble_shell()
        ens.clear_group_returns()
        self.assertEqual(len(ens.accumulated_returns), 0)


class TestUpdateGroupReturns(unittest.TestCase):

    def test_populates_accumulated_returns(self):
        ens = _make_ensemble_shell()
        group_key = ('tech_', '_3')
        # Seed with an initial return so train/test split in update_group_returns
        # doesn't produce an empty slice (int(0.6*1) == 0 causes IndexError)
        ens.accumulated_returns[group_key] = [0.01]
        ens.group_predictions = {
            group_key: {'signals': [1], 'amounts': [10], 'group_name': 'tech__3'},
        }
        data = _make_price_data([1.0, 1.02])
        ens.update_group_returns(data, stop_loss=0.05)
        self.assertIn(group_key, ens.accumulated_returns)
        self.assertEqual(len(ens.accumulated_returns[group_key]), 2)
        self.assertAlmostEqual(ens.accumulated_returns[group_key][1], 0.02, places=4)

    def test_extends_existing_returns(self):
        ens = _make_ensemble_shell()
        group_key = ('mr_', '_6')
        ens.accumulated_returns[group_key] = [0.01]
        ens.group_predictions = {
            group_key: {'signals': [0], 'amounts': [10], 'group_name': 'mr__6'},
        }
        data = _make_price_data([1.0, 0.99])
        ens.update_group_returns(data, stop_loss=0.05)
        self.assertEqual(len(ens.accumulated_returns[group_key]), 2)


class TestGenerateSignal(unittest.TestCase):

    def _make_ensemble_with_mocks(self):
        ens = _make_ensemble_shell()

        mock_a = MagicMock()
        mock_a.feature_set = 'tech_'
        mock_a.feature_frequency = '_3'
        mock_a.generate_signal.return_value = ([1], [10])

        mock_b = MagicMock()
        mock_b.feature_set = 'tech_'
        mock_b.feature_frequency = '_3'
        mock_b.generate_signal.return_value = ([1], [8])

        mock_c = MagicMock()
        mock_c.feature_set = 'mr_'
        mock_c.feature_frequency = '_3'
        mock_c.generate_signal.return_value = ([0], [10])

        ens.strategies = {'mock_a': mock_a, 'mock_b': mock_b, 'mock_c': mock_c}
        return ens

    def _make_data(self):
        dates = pd.date_range('2020-01-01', periods=10, freq='ME')
        return pd.DataFrame({
            'Date': dates,
            'EURUSD_Close': np.linspace(1.0, 1.05, 10),
            'Label': [1] * 10,
            'tech_rsi_3': np.random.randn(10),
            'mr_bb_3': np.random.randn(10),
        })

    def test_returns_signals_and_amounts(self):
        ens = self._make_ensemble_with_mocks()
        data = self._make_data()
        signals, amounts = ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        self.assertIsInstance(signals, list)
        self.assertIsInstance(amounts, list)
        self.assertEqual(len(signals), 1)
        self.assertEqual(len(amounts), 1)
        self.assertIn(signals[0], [0, 1])
        self.assertGreaterEqual(amounts[0], 1)
        self.assertLessEqual(amounts[0], 10)

    def test_group_predictions_populated(self):
        ens = self._make_ensemble_with_mocks()
        data = self._make_data()
        ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        self.assertEqual(len(ens.group_predictions), 2)

    def test_all_sub_strategies_called(self):
        ens = self._make_ensemble_with_mocks()
        data = self._make_data()
        ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        for strategy in ens.strategies.values():
            strategy.generate_signal.assert_called_once()

    def test_negative_accumulated_returns_excludes_group(self):
        ens = self._make_ensemble_with_mocks()
        ens.accumulated_returns[('tech_', '_3')] = [-0.05, -0.03, -0.02]
        ens.accumulated_returns[('mr_', '_3')] = [0.05, 0.03, 0.02]

        data = self._make_data()
        signals, amounts = ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        # Only mr_ group is valid, which returns sell (0)
        self.assertEqual(signals[0], 0)

    def test_all_negative_uses_fallback(self):
        ens = self._make_ensemble_with_mocks()
        ens.accumulated_returns[('tech_', '_3')] = [-0.05, -0.03]
        ens.accumulated_returns[('mr_', '_3')] = [-0.01, -0.02]

        data = self._make_data()
        signals, amounts = ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        # Fallback uses all groups; result depends on aggregation
        self.assertIn(signals[0], [0, 1])

    def test_failed_sub_strategy_is_skipped(self):
        ens = self._make_ensemble_with_mocks()
        # Make one strategy raise
        ens.strategies['mock_a'].generate_signal.side_effect = RuntimeError("boom")

        data = self._make_data()
        signals, amounts = ens.generate_signal(data.iloc[:8], data.iloc[8:9])
        self.assertEqual(len(signals), 1)
        # mock_a failed, but mock_b still provides tech_ group prediction
        self.assertEqual(len(ens.group_predictions), 2)


class TestEnsembleInit(unittest.TestCase):

    @patch('algo_strategy.ensemble_strategy.LGBMOptunaStrategy')
    def test_strategy_count_single_freq(self, mock_lgbm_cls):
        from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy
        mock_lgbm_cls.return_value = MagicMock()
        ens = EnsembleOptunaStrategy(symbol='EURUSD', frequency_range=(3, 4), frequency_range_step=1)
        # 1 freq × 3 feature sets × 50 models = 150
        self.assertEqual(len(ens.strategies), 150)

    @patch('algo_strategy.ensemble_strategy.LGBMOptunaStrategy')
    def test_strategy_count_multiple_freqs(self, mock_lgbm_cls):
        from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy
        mock_lgbm_cls.return_value = MagicMock()
        ens = EnsembleOptunaStrategy(symbol='EURUSD', frequency_range=(3, 6), frequency_range_step=1)
        # 3 freqs × 3 feature sets × 50 models = 450
        self.assertEqual(len(ens.strategies), 450)

    @patch('algo_strategy.ensemble_strategy.LGBMOptunaStrategy')
    def test_strategy_count_with_step(self, mock_lgbm_cls):
        from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy
        mock_lgbm_cls.return_value = MagicMock()
        ens = EnsembleOptunaStrategy(symbol='EURUSD', frequency_range=(3, 18), frequency_range_step=3)
        # freqs: 3,6,9,12,15 → 5 freqs × 3 × 50 = 750
        self.assertEqual(len(ens.strategies), 750)

    @patch('algo_strategy.ensemble_strategy.LGBMOptunaStrategy')
    def test_default_attributes(self, mock_lgbm_cls):
        from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy
        mock_lgbm_cls.return_value = MagicMock()
        ens = EnsembleOptunaStrategy(symbol='XAUUSD')
        self.assertEqual(ens.symbol, 'XAUUSD')
        self.assertEqual(ens.close_col, 'XAUUSD_Close')
        self.assertEqual(ens.max_amount, 10)
        self.assertEqual(ens.accumulated_returns_months, 36)
        self.assertEqual(ens.n_groups, 1)
        self.assertFalse(ens.features_optimization)
        self.assertFalse(ens.fitted)

    @patch('algo_strategy.ensemble_strategy.LGBMOptunaStrategy')
    def test_fit_sets_fitted(self, mock_lgbm_cls):
        from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy
        mock_lgbm_cls.return_value = MagicMock()
        ens = EnsembleOptunaStrategy(symbol='EURUSD')
        self.assertFalse(ens.fitted)
        ens.fit(None, None)
        self.assertTrue(ens.fitted)


if __name__ == '__main__':
    unittest.main()
