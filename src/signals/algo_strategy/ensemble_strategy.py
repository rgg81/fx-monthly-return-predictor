import pandas as pd
import numpy as np
from algo_strategy.pytorch_nn_strategy import PyTorchNeuralNetOptunaStrategy
from strategy import Strategy
from algo_strategy.lgbm_strategy import LGBMOptunaStrategy
from algo_strategy.rf_strategy import RandomForestOptunaStrategy
from algo_strategy.adaboost_strategy import AdaBoostOptunaStrategy
from algo_strategy.histgb_strategy import HistGBOptunaStrategy
from algo_strategy.svc_strategy import SVCOptunaStrategy
from algo_strategy.xgboost_strategy import XGBoostOptunaStrategy
from algo_strategy.catboost_strategy import CatBoostOptunaStrategy
from algo_strategy.ngboost_strategy import NGBoostOptunaStrategy
from algo_strategy.gp_strategy import GaussianProcessOptunaStrategy
from algo_strategy.knn_strategy import KNNOptunaStrategy
from algo_strategy.logistic_strategy import LogisticRegressionOptunaStrategy
from algo_strategy.nb_strategy import GaussianNBOptunaStrategy
from algo_strategy.mlp_strategy import MLPOptunaStrategy
import warnings
import quantstats as qs

warnings.filterwarnings('ignore')

class EnsembleOptunaStrategy(Strategy):
    """
    An ensemble strategy that combines multiple Optuna-optimized strategies.
    Uses LGBMOptunaStrategy, RandomForestOptunaStrategy, AdaBoostOptunaStrategy,
    HistGBOptunaStrategy, XGBoostOptunaStrategy, CatBoostOptunaStrategy, 
    NGBoostOptunaStrategy, and GaussianProcessOptunaStrategy.
    """
    
    def __init__(self, max_amount=10, feature_set=None, symbol="USDJPY", stop_loss=0.1, features_optimization=False, frequency_range=(3, 4), frequency_range_step=1):
        self.max_amount = max_amount
        self.strategies = {}
        self.fitted = False
        self.feature_set = feature_set
        self.symbol = symbol
        self.stop_loss = stop_loss
        self.close_col = f"{symbol}_Close"
        self.features_optimization = features_optimization
        self.accumulated_returns_months = 36  # Number of months to track for accumulated returns
        self.n_groups = 1
        self.frequency_range = frequency_range
        self.frequency_range_step = frequency_range_step
        
        # Track accumulated returns per group (feature_set, feature_frequency)
        self.accumulated_returns = {}  # Key: (feature_set, feature_frequency), Value: cumulative return
        
        #for freq in range(3, 6):
        for freq in range(self.frequency_range[0], self.frequency_range[1], self.frequency_range_step):     
            for i in range(0, 10):
                self.strategies[f'LGBM_regime_{freq}_{i}'] = CatBoostOptunaStrategy(feature_set="regime_", symbol=symbol, n_trials=10, feature_frequency=f"_{freq}")
            for i in range(0, 10):
                self.strategies[f'LGBM_tech_{freq}_{i}'] = CatBoostOptunaStrategy(feature_set="tech_", symbol=symbol, n_trials=10, feature_frequency=f"_{freq}")
            for i in range(0, 10):
                self.strategies[f'LGBM_mr_{freq}_{i}'] = CatBoostOptunaStrategy(feature_set="mr_", symbol=symbol, n_trials=10, feature_frequency=f"_{freq}")
        
        print(f"Ensemble Strategy initialized with {len(self.strategies)} strategies")

    def fit(self, X, y):
        """
        Ensemble doesn't need training - delegates to individual strategies.
        This method is required by the Strategy base class.
        """
        self.fitted = True
        pass

    def _aggregate_predictions(self, predictions, group_name=""):
        """Aggregate predictions using weighted voting based on amounts"""
        if not predictions:
            print(f"No valid predictions for {group_name}, returning default")
            return [0], [self.max_amount]
        
        print(f"\nAggregating {len(predictions)} predictions for {group_name}:")

        total_signals = len(list(predictions.items())[0][1]['signals'])
        ensemble_signals = []
        ensemble_amounts = []
        
        for i in range(total_signals):
            total_weight = 0
            weighted_signal_sum = 0
        
            for name, pred in predictions.items():
                signal = pred['signals'][i]
                amount = pred['amounts'][i]
                
                if signal == 1:
                    total_weight += self.max_amount
                    weighted_signal_sum += amount
                else:
                    total_weight += self.max_amount
                    weighted_signal_sum -= amount
            
            # Calculate ensemble signal and amount
            if total_weight == 0:
                ensemble_signal = 0
                ensemble_amount = self.max_amount
            else:
                normalized_signal = weighted_signal_sum / total_weight
                
                if normalized_signal > 0:
                    ensemble_signal = 1
                    ensemble_amount = min(self.max_amount, abs(normalized_signal) * self.max_amount)
                else:
                    ensemble_signal = 0
                    ensemble_amount = min(self.max_amount, abs(normalized_signal) * self.max_amount)
                
                ensemble_amount = max(1, min(self.max_amount, int(ensemble_amount)))
            
            print(f"  Signal {i}: Weighted Sum={weighted_signal_sum:.2f}, Total Weight={total_weight}, Final Signal={ensemble_signal}, Amount={ensemble_amount}")
            ensemble_signals.append(ensemble_signal)
            ensemble_amounts.append(ensemble_amount)
        
        return ensemble_signals, ensemble_amounts
    
    def _calculate_profit_loss(self, signals, amounts, data, stop_loss):
        """Calculate profit/loss for given signals and amounts"""
        profit_loss_list = []
        index_next = 1
        
        for signal, amount in zip(signals, amounts):
            current_data = data.iloc[index_next - 1]
            profit_loss = 0.0
            if signal == 1:  # Buy signal
                if index_next < len(data):
                    next_close = data.iloc[index_next][self.close_col]
                    # percentage change
                    profit_loss = ((next_close - current_data[self.close_col]) / current_data[self.close_col]) 
                    profit_loss = max(min(profit_loss, stop_loss), -stop_loss) * (amount / self.max_amount)
            else:
                if index_next < len(data):
                    next_close = data.iloc[index_next][self.close_col]
                    # percentage change
                    profit_loss = ((current_data[self.close_col] - next_close) / current_data[self.close_col]) 
                    profit_loss = max(min(profit_loss, stop_loss), -stop_loss) * (amount / self.max_amount)                        
            profit_loss_list.append(profit_loss)
            print(f"=== Date: {current_data['Date']}, Label: {current_data['Label']} Signal: {signal}, Amount: {amount}, Return: {profit_loss}, current close: {current_data[self.close_col]} next close: {next_close if index_next < len(data) else 'N/A'} ===", flush=True)
            index_next += 1
        
        return profit_loss_list 

    def generate_signal(self, past_data, current_data):
        """Generate ensemble signal from all strategies with grouped performance tracking"""
        print(f"\n{'='*80}")
        print(f"--- Ensemble Prediction (Month: {current_data.iloc[0]['Date']}) ---")
        print(f"{'='*80}")
        
        # Include necessary columns
        if self.feature_set is None:
            feature_columns = past_data.columns
        else:
            feature_columns = [col for col in past_data.columns if col.startswith(self.feature_set)]
            feature_columns.extend(['Label', 'Date', self.close_col])
        
        print(f"Feature columns for prediction: {len(feature_columns)} columns")

        # STEP 1: Group strategies by feature_set and feature_frequency
        grouped_strategies = {}
        
        for name, strategy in self.strategies.items():
            group_key = (strategy.feature_set, strategy.feature_frequency)
            if group_key not in grouped_strategies:
                grouped_strategies[group_key] = {}
            grouped_strategies[group_key][name] = strategy
        
        print(f"\nGrouped into {len(grouped_strategies)} groups (feature_set × feature_frequency)")
        
        # STEP 2 & 3: Get predictions for each group and aggregate within group
        self.group_predictions = {}  # Stores aggregated predictions per group
        
        for group_key, strategies_in_group in grouped_strategies.items():
            feature_set, feature_freq = group_key
            group_name = f"{feature_set}{feature_freq}"
            
            print(f"\n{'='*60}")
            print(f"Processing Group: {group_name} ({len(strategies_in_group)} models)")
            print(f"{'='*60}")
            
            # Get predictions from all strategies in this group
            strategy_predictions = {}
            
            for name, strategy in strategies_in_group.items():
                try:
                    signals, amounts = strategy.generate_signal(past_data[feature_columns], current_data[feature_columns])
                    strategy_predictions[name] = {'signals': signals, 'amounts': amounts}
                except Exception as e:
                    print(f"Failed to get prediction from {name}: {e}")
                    continue
            
            if not strategy_predictions:
                print(f"No valid predictions for group {group_name}, skipping")
                continue
            
            # Aggregate predictions within this group (100 models)
            group_signals, group_amounts = self._aggregate_predictions(strategy_predictions, group_name)
            
            # Store for later use
            self.group_predictions[group_key] = {
                'signals': group_signals,
                'amounts': group_amounts,
                'group_name': group_name
            }
        
        # STEP 4: Calculate profit/loss for each group and check accumulated returns
        print(f"\n{'='*80}")
        print(f"Performance Check and Filtering")
        print(f"{'='*80}")
        
        valid_group_predictions = {}  # Groups that pass the accumulated return check
        
        for group_key, pred_data in self.group_predictions.items():
            feature_set, feature_freq = group_key
            group_name = pred_data['group_name']
            
            # Get previous accumulated return for this group
            prev_accumulated_return = self.accumulated_returns.get(group_key, [0.0])

            last_months_returns = prev_accumulated_return[-self.accumulated_returns_months:] if len(prev_accumulated_return) >= self.accumulated_returns_months else prev_accumulated_return

            #train_max = int(0.6*len(last_months_returns))
            #last_months_returns_train = last_months_returns[:train_max]
            #last_months_returns_test = last_months_returns[train_max:]
            last_months_returns_train = last_months_returns
            last_months_returns_test = last_months_returns

            cum_returns_train = qs.stats.compsum(pd.Series(last_months_returns_train))
            cum_return_value_train = cum_returns_train.iloc[-1] if isinstance(cum_returns_train, pd.Series) else float(cum_returns_train)
            cum_returns_test = qs.stats.compsum(pd.Series(last_months_returns_test))
            cum_return_value_test = cum_returns_test.iloc[-1] if isinstance(cum_returns_test, pd.Series) else float(cum_returns_test)

            # calculate accumalted return based all monthly returns represented in this variable prev_accumulated_return
            cum_returns_months = qs.stats.compsum(pd.Series(last_months_returns))
            # Get the last value as float - handle both Series and scalar returns
            cum_return_value_months = cum_returns_months.iloc[-1] if isinstance(cum_returns_months, pd.Series) else float(cum_returns_months)

            
            print(f"\nGroup {group_name}:")
            print(f" all returns:{prev_accumulated_return} months: {cum_return_value_months:.4f} Acc return train: {cum_return_value_train:.4f} Acc return test: {cum_return_value_test:.4f} Previous accumulated return {self.accumulated_returns_months}")
            
            # Calculate P&L for current prediction
            # We need to simulate what would happen with current prediction
            # Note: This assumes we have access to next period data for evaluation
            # In real trading, this would be calculated after the fact
            
            # For now, we'll use a placeholder - in practice, you'd update this after seeing results
            # This is a forward-looking check, so we include if previous performance was positive
            
            if cum_return_value_months >= 0:
                print(f"  ✅ Including group {group_name} (positive accumulated return)")
                valid_group_predictions[group_key] = {
                    'pred_data': pred_data,
                    'cum_return': cum_return_value_months,
                    'cum_return_train': cum_return_value_train,
                    'cum_return_test': cum_return_value_test
                }
            else:
                print(f"  ❌ Excluding group {group_name} (negative accumulated return)")
        
        # STEP 5: Final aggregation from valid groups only
        print(f"\n{'='*80}")
        print(f"Final Ensemble Aggregation")
        print(f"{'='*80}")
        print(f"Positive groups: {len(valid_group_predictions)} out of {len(self.group_predictions)} groups")
        
        if not valid_group_predictions:
            print("⚠️ No valid groups with positive returns, using all groups as fallback")
            # Use all groups as fallback
            for group_key, pred_data in self.group_predictions.items():
                prev_accumulated_return = self.accumulated_returns.get(group_key, [0.0])
                train_max = int(0.6*len(last_months_returns))
                last_months_returns_train = last_months_returns[:train_max]
                last_months_returns_test = last_months_returns[train_max:]

                cum_returns_train = qs.stats.compsum(pd.Series(last_months_returns_train))
                cum_return_value_train = cum_returns_train.iloc[-1] if isinstance(cum_returns_train, pd.Series) else float(cum_returns_train)
                cum_returns_test = qs.stats.compsum(pd.Series(last_months_returns_test))
                cum_return_value_test = cum_returns_test.iloc[-1] if isinstance(cum_returns_test, pd.Series) else float(cum_returns_test)
                
                cum_returns = qs.stats.compsum(pd.Series(prev_accumulated_return))
                cum_return_value = cum_returns.iloc[-1] if isinstance(cum_returns, pd.Series) else float(cum_returns)
                valid_group_predictions[group_key] = {
                    'pred_data': pred_data,
                    'cum_return': cum_return_value,
                    'cum_return_train': cum_return_value_train,
                    'cum_return_test': cum_return_value_test
                }
        
        # Sort groups by cumulative return (descending) and take top 3
        #sorted_groups_test = sorted(valid_group_predictions.items(), key=lambda x: x[1]['cum_return_test'], reverse=True)
        sorted_groups_all = sorted(valid_group_predictions.items(), key=lambda x: x[1]['cum_return'], reverse=True)
        #sorted_groups_train = sorted(valid_group_predictions.items(), key=lambda x: x[1]['cum_return_train'], reverse=True)
        
        # Combine the best from all three sorts
        combined_top_groups = {}
        for group_key, group_info in sorted_groups_all[:self.n_groups]:
            combined_top_groups[group_key] = group_info
        #for group_key, group_info in sorted_groups_train[:self.n_groups]:
        #    combined_top_groups[group_key] = group_info
        #for group_key, group_info in sorted_groups_test[:self.n_groups]:
        #    combined_top_groups[group_key] = group_info
        
        top_n_groups = list(combined_top_groups.items())[:self.n_groups]
        
        print(f"Top {self.n_groups} groups selected:")
        for group_key, group_info in top_n_groups:
            print(f"  - {group_info['pred_data']['group_name']}: cum_return_all = {group_info['cum_return']:.4f} cum_return_train = {group_info['cum_return_train']:.4f} cum_return_test = {group_info['cum_return_test']:.4f}")
        
        # Aggregate across top 3 groups only
        final_predictions = {}
        for group_key, group_info in top_n_groups:
            pred_data = group_info['pred_data']
            final_predictions[pred_data['group_name']] = {
                'signals': pred_data['signals'],
                'amounts': pred_data['amounts']
            }
        
        ensemble_signals, ensemble_amounts = self._aggregate_predictions(final_predictions, "FINAL ENSEMBLE")
        
        print(f"\n{'='*80}")
        print(f"Final Prediction: Signal={ensemble_signals}, Amount={ensemble_amounts}")
        print(f"{'='*80}\n")

        return ensemble_signals, ensemble_amounts
    
    def update_group_returns(self, data, stop_loss):
        """Update accumulated returns for a specific group after observing results"""
        for group_key, pred_data in self.group_predictions.items():
            feature_set, feature_freq = group_key
            group_name = pred_data['group_name']
            signals = pred_data['signals']
            amounts = pred_data['amounts']
            profit_losses = self._calculate_profit_loss(signals, amounts, data, stop_loss)
            if group_key not in self.accumulated_returns:
                self.accumulated_returns[group_key] = []
            self.accumulated_returns[group_key].extend(profit_losses)
            # print information here
            cum_return = qs.stats.compsum(pd.Series(self.accumulated_returns[group_key]))
            last_months_returns = self.accumulated_returns[group_key][-self.accumulated_returns_months:] if len(self.accumulated_returns[group_key]) >= self.accumulated_returns_months else self.accumulated_returns[group_key]

            train_max = int(0.6*len(last_months_returns))
            last_months_returns_train = last_months_returns[:train_max]
            last_months_returns_test = last_months_returns[train_max:]
            cum_return_train = qs.stats.compsum(pd.Series(last_months_returns_train))
            cum_return_test = qs.stats.compsum(pd.Series(last_months_returns_test))

            cum_return_months = qs.stats.compsum(pd.Series(last_months_returns))
            
            cum_return_value_months = cum_return_months.iloc[-1] if isinstance(cum_return_months, pd.Series) else float(cum_return_months)
            cum_return_value = cum_return.iloc[-1] if isinstance(cum_return, pd.Series) else float(cum_return)
            cum_return_value_train = cum_return_train.iloc[-1] if isinstance(cum_return_train, pd.Series) else float(cum_return_train)
            cum_return_value_test = cum_return_test.iloc[-1] if isinstance(cum_return_test, pd.Series) else float(cum_return_test)

            print(f"Updated Group name = {group_name} Acc returns: {cum_return_value:.4f}  months returns: {cum_return_value_months:.4f}  train returns: {cum_return_value_train:.4f} test returns: {cum_return_value_test:.4f} signals: {signals} amounts: {amounts} Returns monthly = {self.accumulated_returns[group_key]} {self.accumulated_returns_months}", flush=True)
    
    def clear_group_returns(self):
        """Clear accumulated returns for all groups"""
        for group_key in self.accumulated_returns.keys():
            self.accumulated_returns[group_key] = [0.0]
        print("Cleared accumulated returns for all groups.")