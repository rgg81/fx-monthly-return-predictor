import pandas as pd
from strategy import RandomStrategy
import quantstats as qs

class Backtest:
    def __init__(self, strategy, max_amount=10, stop_loss=0.015, close_col='Close', min_history=154, start_year=2016):
        """
        Initialize the backtest.
        """
        self.strategy = strategy
        self.max_amount = max_amount
        self.stop_loss = stop_loss
        self.close_col = close_col
        self.min_history = min_history
        # Convert start_year to datetime for date comparison
        if isinstance(start_year, int):
            self.start_year = pd.to_datetime(f'{start_year}-01-01')
        elif isinstance(start_year, str):
            self.start_year = pd.to_datetime(start_year)
        else:
            self.start_year = pd.to_datetime(start_year)

    def run(self, data):
        """
        Run the backtest on the given data.
        :param data: DataFrame with columns ['Date', 'Close', 'Feature1', 'Feature2', ...]
        :return: DataFrame with backtest results
        """
        results = []
        data = data.sort_values('Date')  # Ensure data is sorted by date
        step_size = 6  # You can adjust the step size if needed
        start_i_range = 0

        for i in range(len(data)):
            current_data_frame = data.iloc[i:i+1]
            past_data = data.iloc[:i+1]
            if past_data.empty or i < self.min_history or current_data_frame.empty:  # Ensure we have enough past data for the strategy
            # Skip the first row since there's no past data
                continue

            if current_data_frame.iloc[0]['Date'] < self.start_year:
                continue
            start_i_range = i
            break

        for i in range(start_i_range, len(data), step_size):
            current_step = i + step_size
            current_data_frame = data.iloc[i:current_step]
            current_data_frame_plus_next = data.iloc[i:current_step+1]
            past_data = data.iloc[:current_step]


            signals, amounts = self.strategy.generate_signal(past_data, current_data_frame)
            if signals is None: continue
            index_next = i + 1
            for signal, amount in zip(signals, amounts):
                profit_loss = 0.0
                current_data = data.iloc[index_next - 1]
                if signal == 1:  # Buy signal
                    if index_next < len(data):
                        next_close = data.iloc[index_next][self.close_col]
                        # percentage change
                        profit_loss = ((next_close - current_data[self.close_col]) / current_data[self.close_col]) 
                        profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss) * (amount / self.max_amount)
                else:
                    if index_next < len(data):
                        next_close = data.iloc[index_next][self.close_col]
                        # percentage change
                        profit_loss = ((current_data[self.close_col] - next_close) / current_data[self.close_col]) 
                        profit_loss = max(min(profit_loss, self.stop_loss), -self.stop_loss) * (amount / self.max_amount)
                result = {
                    'Date': current_data['Date'],
                    'Signal': signal,
                    'Amount': amount,
                    'Return': profit_loss
                }
                if self.strategy.features_optimization:
                    print(f"*** Date: {current_data['Date']}, Label: {current_data['Label']} Signal: {signal}, Amount: {amount}, Return: {profit_loss}, current close: {current_data[self.close_col]} next close: {next_close if index_next < len(data) else 'N/A'} ***", flush=True)
                else:
                    print(f"--- Date: {current_data['Date']}, Label: {current_data['Label']} Signal: {signal}, Amount: {amount}, Return: {profit_loss}, current close: {current_data[self.close_col]} next close: {next_close if index_next < len(data) else 'N/A'} ---", flush=True)
                results.append(result)
                index_next += 1
            # check if strategy is ensemble and update accumulated returns
            if hasattr(self.strategy, 'update_group_returns'):
                self.strategy.update_group_returns(current_data_frame_plus_next, self.stop_loss)

            # Print current accumulated series of returns
            last_months_returns = [item['Return'] for item in results]
            cum_returns_months = qs.stats.compsum(pd.Series(last_months_returns))
            print(f"Accumulated Returns (Last {self.strategy.accumulated_returns_months} Months): {cum_returns_months}")

        return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=12, freq='M'),
        'Close': [100, 105, 102, 110, 120, 115, 125, 130, 128, 135, 140, 145],
        'Feature1': range(12),
        'Feature2': range(12, 24)
    })
    
    # Test with the RandomStrategy
    random_strategy = RandomStrategy()
    random_backtest = Backtest(random_strategy)
    random_results = random_backtest.run(data)
    print("\nRandom Strategy Results:")
    print(random_results)