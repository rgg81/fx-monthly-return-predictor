import datetime
import os
import subprocess
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd

# Configuration
LIVE_MODE = True
DOWNLOAD_DIR = "download"
HISTORICAL_DAYS = 9200
CHUNK_DAYS = 300
SYMBOLS = ["EURUSD", "USDJPY", "EURJPY", "GBPUSD", "AUDUSD", "XAUUSD"]


def ensure_download_directory():
    """Ensure the download directory exists."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)


def download_symbol_data(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """
    Download symbol data for a given date range using dukascopy-node.
    """
    instrument = symbol.lower()
    output_file = f"{instrument}"

    subprocess.run([
        "npx", "dukascopy-node",
        "-i", instrument,
        "-from", start_date.strftime('%Y-%m-%d'),
        "-to", end_date.strftime('%Y-%m-%d'),
        "-t", "h1",
        "-f", "csv",
        "-fn", output_file
    ], check=True)

    try:
        df = pd.read_csv(f"download/{output_file}.csv")
        return df
    except Exception as ex:
        print(f"Error reading downloaded data for {symbol}: {ex}")
        return pd.DataFrame()


def download_symbol(symbol: str):
    """
    Download historical data for a symbol in chunks and save it to a CSV file.
    """
    ensure_download_directory()

    result = deque()
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=HISTORICAL_DAYS)

    print(f"Downloading data for {symbol}...")

    while True:
        chunk_end_date = end_date
        chunk_start_date = chunk_end_date - datetime.timedelta(days=CHUNK_DAYS)

        if chunk_start_date < start_date:
            chunk_start_date = start_date

        df_chunk = download_symbol_data(symbol, chunk_start_date, chunk_end_date)

        if df_chunk.empty:
            print(f"No more data available for {symbol}.")
            break

        result.appendleft(df_chunk.to_numpy())

        if chunk_start_date <= start_date:
            print(f"Reached the start date for {symbol}.")
            break

        end_date = chunk_start_date

    if not result:
        print(f"No data downloaded for {symbol}.")
        return

    # Combine all chunks into a single DataFrame
    combined_data = np.concatenate(result, axis=0)
    df = pd.DataFrame(combined_data, columns=["Date", f"{symbol}_Open", f"{symbol}_High", f"{symbol}_Low", f"{symbol}_Close"])

    # Process the DataFrame
    df["Date"] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index("Date", inplace=True)
    df = df.resample("ME").last()  # Resample to monthly data
    # Save to CSV
    output_file = f"{symbol}.csv"
    df.to_csv(output_file)
    print(f"Data for {symbol} saved to {output_file}.")
    return df


def get_all_symbols_data():
    """
    Download data for all symbols and return a list of DataFrames.
    """
    # List of symbols to download
    with Pool(processes=1 if LIVE_MODE else len(SYMBOLS)) as pool:
        results = pool.map(download_symbol, SYMBOLS)
    return results


if __name__ == '__main__':

    # Use multiprocessing to download data for all symbols
    dataframes = get_all_symbols_data()

    # Print sample data
    for symbol, df in zip(SYMBOLS, dataframes):
        print(f"Data for {symbol}:")
        print(df.tail())
