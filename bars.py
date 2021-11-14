import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

# expects a numpy array with trades
# each trade is composed of: [time, price, quantity]
def generate_volumebars(df, frequency=1e9):

    # times = trades[:,0]
    # prices = trades[:,1]
    # volumes = trades[:,2]

    times = df.index.to_numpy()
    prices = data.Close.to_numpy()
    volumes = data.Volume.to_numpy()

    ans = np.zeros(shape=(len(prices), 6))
    date_list = []
    candle_counter = 0
    dollars = 0
    lasti = 0
    for i in range(len(prices)):
        dollars += volumes[i]*prices[i]
        if dollars >= frequency:
            ans[candle_counter][0] = times[i]                          # time
            date_list.append(times[i])
            ans[candle_counter][1] = prices[lasti]                     # open
            ans[candle_counter][2] = np.max(prices[lasti:i+1])         # high
            ans[candle_counter][3] = np.min(prices[lasti:i+1])         # low
            ans[candle_counter][4] = prices[i]                         # close
            ans[candle_counter][5] = np.sum(volumes[lasti:i+1])        # volume
            candle_counter += 1
            lasti = i+1
            vol = 0

    df = pd.DataFrame(ans[:candle_counter], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], index = date_list)
    return df


def generate_tickbars(df, frequency=100):
    times = df.index.to_numpy()
    prices = data.Close.to_numpy()
    volumes = data.Volume.to_numpy()

    res = np.zeros(shape=(len(range(frequency, len(prices), frequency)), 6))
    date_list = []
    it = 0
    for i in range(frequency, len(prices), frequency):
        res[it][0] = times[i-1]                        # time
        date_list.append(times[i])
        res[it][1] = prices[i-frequency]               # open
        res[it][2] = np.max(prices[i-frequency:i])     # high
        res[it][3] = np.min(prices[i-frequency:i])     # low
        res[it][4] = prices[i-1]                       # close
        res[it][5] = np.sum(volumes[i-frequency:i])    # volume
        it += 1

    df = pd.DataFrame(res, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], index = date_list)
    return df


tick = 'AAPL'

# raw_folder = Path.cwd() / 'data' / 'raw'
# data = pd.read_pickle(raw_folder / f'{tick}.pkl')


data = yf.Ticker(tick).history(period='max', interval='1d')


data.index = pd.to_datetime(data.index)

# print(data), quit()
# print(data.Close.to_numpy())
# print(data.index.to_numpy())

ans = generate_tickbars(data)
# ans = generate_volumebars(data)

print(ans)
print(data)