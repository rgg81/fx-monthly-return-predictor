import pandas as pd
import numpy as np
import quantstats as qs
from backtest import Backtest
from algo_strategy.ensemble_strategy import EnsembleOptunaStrategy

qs.extend_pandas()

def analyze_strategy_performance(strategy_results, strategy_name="Trading Strategy", symbol=None):
    result_array_series = strategy_results.set_index("Date")["Return"]
    cum_returns = qs.stats.compsum(result_array_series)
    
    output = 'stats.html' if symbol is None else f'stats_{symbol}.html'
    qs.reports.html(result_array_series, output=output, title=f'{strategy_name} Performance Report')
    
    print(f"\n----- {strategy_name} Performance Metrics -----")
    print(f"Cumulative Return: {cum_returns[-1]:.2%}")
    print(f"CAGR: {qs.stats.cagr(result_array_series):.2%}")
    print(f"Sharpe Ratio: {qs.stats.sharpe(result_array_series):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(result_array_series):.2%}")

    return cum_returns[-1]
    

def load_features_data(symbol):
    data = pd.read_csv(f"{symbol}.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index("Date", inplace=True)
    
    features_technical = pd.read_csv(f"technical_indicators_features_{symbol}.csv", parse_dates=["Date"])
    features_technical.set_index("Date", inplace=True)
    features_technical.rename(columns=lambda x: f"tech_{x}" if x not in ["Date", "Label", f"{symbol}_Close"] else x, inplace=True)
    data = data.join(features_technical, how="inner")

    features_mean_reversion = pd.read_csv(f"mean_reversion_features_{symbol}.csv", parse_dates=["Date"])
    features_mean_reversion.set_index("Date", inplace=True)
    features_mean_reversion.rename(columns=lambda x: f"mr_{x}" if x not in ["Date", "Label", f"{symbol}_Close"] else x, inplace=True)
    data = data.join(features_mean_reversion, how="inner")

    features_regime = pd.read_csv(f"regime_detection_features_{symbol}.csv", parse_dates=["Date"])
    features_regime.set_index("Date", inplace=True)
    features_regime.rename(columns=lambda x: f"regime_{x}" if x not in ["Date", "Label", f"{symbol}_Close"] else x, inplace=True)
    data = data.join(features_regime, how="inner")

    data.reset_index(inplace=True)
    return data


def calculate_adaptive_portfolio_returns(pair_results_dict, weights=None):
    if weights is None:
        weights = {pair: 1.0 for pair in pair_results_dict.keys()}
    
    combined_df = pd.DataFrame({'Date': list(pair_results_dict.values())[0]['Date']})
    for pair_name, result_df in pair_results_dict.items():
        combined_df[pair_name] = result_df['Return'].values
    
    pair_cumulative_returns = {
        pair: qs.stats.compsum(pd.Series(combined_df[pair].values))
        for pair in pair_results_dict.keys()
    }
    
    adaptive_returns = []
    
    for idx in range(len(combined_df)):
        if idx == 0:
            profitable_pairs = list(pair_results_dict.keys())
        else:
            accumulated_returns = {pair: pair_cumulative_returns[pair].iloc[idx-1] for pair in pair_results_dict.keys()}
            profitable_pairs = [pair for pair, acc_return in accumulated_returns.items() if acc_return > 0]
            
        if len(profitable_pairs) == 0:
            adaptive_returns.append(0.0)
            print(f"Month {idx}: No profitable pairs, return set to 0.0", flush=True)
            continue
        
        weighted_sum = sum(combined_df[pair].iloc[idx] * weights[pair] for pair in profitable_pairs)
        total_weight = sum(weights[pair] for pair in profitable_pairs)
        current_return = weighted_sum / total_weight
        
        adaptive_returns.append(current_return)
        print(f"Month {idx}: Profitable pairs: {profitable_pairs}, Current return: {current_return:.4f}", flush=True)
    
    return pd.DataFrame({'Date': combined_df['Date'], 'Return': adaptive_returns})


def run_ensemble_models(symbol, stop_loss, start_year='2014-06-01', min_history=100):
    data = load_features_data(symbol)
    
    strategy_model1 = EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3)
    backtest_model1 = Backtest(strategy_model1, close_col=f'{symbol}_Close', stop_loss=stop_loss, start_year=start_year, min_history=min_history)
    result_model1 = backtest_model1.run(data)
    analyze_strategy_performance(result_model1, strategy_name=f"{symbol} Model1", symbol=f"{symbol}_model1")
    
    strategy_model2 = EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1)
    backtest_model2 = Backtest(strategy_model2, close_col=f'{symbol}_Close', stop_loss=stop_loss, start_year=start_year, min_history=min_history)
    result_model2 = backtest_model2.run(data)
    analyze_strategy_performance(result_model2, strategy_name=f"{symbol} Model2", symbol=f"{symbol}_model2")
    
    result_combined = pd.DataFrame({
        'Date': result_model1['Date'],
        'Signal': result_model1['Signal'],
        'Amount': result_model1['Amount'],
        'Return': (result_model1['Return'] + result_model2['Return']) / 2
    })
    analyze_strategy_performance(result_combined, strategy_name=f"{symbol} Combined", symbol=f"{symbol}_combined")
    
    return result_combined


if __name__ == "__main__":
    pair_configs = {
        'EURUSD': {'stop_loss': 0.015},
        'USDJPY': {'stop_loss': 0.025},
        'EURJPY': {'stop_loss': 0.025},
        'GBPUSD': {'stop_loss': 0.020},
        'AUDUSD': {'stop_loss': 0.025},
        'XAUUSD': {'stop_loss': 0.025}
    }
    
    pair_weights = {
        'EURUSD': 2.5,
        'USDJPY': 1.5,
        'EURJPY': 1.5,
        'GBPUSD': 2.0,
        'AUDUSD': 1.5,
        'XAUUSD': 1.5
    }
    
    pair_results = {}
    for symbol, config in pair_configs.items():
        print(f"\n{'='*80}\nProcessing {symbol}\n{'='*80}")
        pair_results[symbol] = run_ensemble_models(symbol, config['stop_loss'])
    
    adaptive_portfolio = calculate_adaptive_portfolio_returns(pair_results, weights=pair_weights)
    
    weighted_average = pd.DataFrame({
        'Date': pair_results['EURUSD']['Date'],
        'Return': sum(pair_results[pair]['Return'] * pair_weights[pair] for pair in pair_configs.keys()) / sum(pair_weights.values())
    })
    analyze_strategy_performance(weighted_average, strategy_name="Weighted Average Portfolio", symbol="simple_average")
    
    penalty_rate = 0.05
    weighted_average_with_costs = weighted_average.copy()
    weighted_average_with_costs['Return'] = weighted_average_with_costs['Return'].apply(
        lambda x: x * (1 - penalty_rate) if x > 0 else x * (1 + penalty_rate)
    )
    analyze_strategy_performance(weighted_average_with_costs, strategy_name="Portfolio with Costs", symbol="accounting_for_costs")
    
    analyze_strategy_performance(adaptive_portfolio, strategy_name="Adaptive Portfolio", symbol="adaptive_portfolio")
