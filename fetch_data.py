import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import glob


class Fetch:

    def __init__(self, tick):
        
        self.tick = tick
        self.data_folder = Path.cwd() / 'data' / 'raw_intraday'
        self.ticker = yf.Ticker(self.tick)

        self._fetch()
        self._save()


    def _fetch(self):
        self.df = self.ticker.history(start=datetime.today()-timedelta(days=730), end=datetime.today(), period='max', interval='1h')
        print(self.df)

    def _save(self):
        self.df.to_pickle(self.data_folder / f"{self.tick}.pkl")




if __name__ == '__main__':
    tickers = ['ETH-USD']

    for tick in tickers:
        Fetch(tick=tick)
    
    print(f'---> EOL: {__file__}')