import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import glob
import pandas as pd
from tqdm import tqdm


class Fetch:

    def __init__(self, tick):
        
        self.tick = tick
        self.data_folder = Path.cwd() / 'data' / 'raw'
        self.ticker = yf.Ticker(self.tick)

        self._fetch()
        self._save()


    def _fetch(self):
        # print('TICK:', self.tick)
        # print('START:', datetime.today()-timedelta(days=60))
        # self.df = self.ticker.history(start=datetime.today()-timedelta(days=60), end=datetime.today(), period='max', interval='5m')
        self.df = self.ticker.history(period='max', interval='1d')

    def _save(self):
        self.df.to_pickle(self.data_folder / f"{self.tick}.pkl")




if __name__ == '__main__':
    # tickers = ['ERIC-B.ST', 'VOLV-B.ST', 'AZN.ST', 'SAND.ST', 'TEL2-B.ST', 'HM-B.ST', 'SEB-A.ST', 'INVE-A.ST', 'LUNE.ST']
    # tickers = ['ALFA.ST']
    # tickers = ['AAPL', 'AMZN', 'AXP', 'BA', 'BAC', 'BBBY', 'CAKE', 'CSCO', 'DELL', 'DIS', 'DOCU', 'F', 'GOOG', 'JNJ', 'JPM', 'KO', 'KSS', 'MA', 'MMM', 'NFLX', 'PEP', 'PG', 'PYPL', 'ROKU', 'SBUX', 'SHOP', 'T']
    # tickers = ['FB']

    tickers = pd.read_csv('tickers.csv').A.to_list()


    for tick in tqdm(tickers):
        try:
            Fetch(tick=tick)
        except:
            print(f'Not able to fetch {tick}')
    
    print(f'---> EOL: {__file__}')