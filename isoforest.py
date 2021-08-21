import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pandas as pd
import pandas_ta as ta
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import zscore
import glob
from sklearn.decomposition import PCA


raw_folder = Path.cwd() / 'data' / 'raw_intraday'
# tick = 'AAPL'
# tick2 = 'JPM'
all_ticks = [x.stem for x in raw_folder.glob('*.pkl')]
tick = random.choice(all_ticks)

split = 1000
df = pd.read_pickle(raw_folder / f'{tick}.pkl')[0:split]
# df_test = pd.read_csv(raw_folder / f'{tick_test}.csv')
# df_test = pd.read_csv(raw_folder / f'{tick}.csv')[split::]

df.ta.ema(length=12, append=True)
df.ta.ema(length=24, append=True)
df['ema_diff'] = (df.EMA_12 - df.EMA_24).pct_change()

df.ta.macd(fast=12, slow=26, append=True)
df.ta.rsi(append=True)
df['RSI_SMA'] = df.RSI_14.rolling(window=7).mean()
df['RSI_SMA_diff'] = (df.RSI_14 - df.RSI_SMA)
df['close_pct'] = df.close.pct_change().shift(periods=1)

df['RSI_shift1'] = df.RSI_14.shift(periods=1) / 1
df['RSI_shift2'] = df.RSI_14.shift(periods=2) / 2
df['RSI_shift3'] = df.RSI_14.shift(periods=3) / 3
df['RSI_shift4'] = df.RSI_14.shift(periods=4) / 4
df['RSI_shift5'] = df.RSI_14.shift(periods=5) / 5
df['RSI_shift6'] = df.RSI_14.shift(periods=6) / 6
df['RSI_shift7'] = df.RSI_14.shift(periods=7) / 7


df['close_shift2'] = df.close.shift(periods=2) / 2
df['close_shift1'] = df.close.shift(periods=1) / 1
df['close_shift3'] = df.close.shift(periods=3) / 3
df['close_shift4'] = df.close.shift(periods=4) / 4
df['close_shift5'] = df.close.shift(periods=5) / 5
df['close_shift6'] = df.close.shift(periods=6) / 6
df['close_shift7'] = df.close.shift(periods=7) / 7

bband_length = 100
bband = df.copy().ta.bbands(length=bband_length)
bband['hlc'] = df.copy().ta.hlc3()
bbu_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBU_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])
bbl_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBL_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])
df['BBU_signal'] = bbu_signal
df['BBL_signal'] = bbl_signal

df.dropna(inplace=True)
df.reset_index(inplace=True)

# print(df), quit()



def process(df, start, mode):

    if mode=='train':
        # if typ=='peak':
        #     # df = df[(df.RSI_14>60)]
        #     df = df[(df.MACDh_12_26_9<0)]
        # elif typ=='valley':
        #     # df = df[(df.RSI_14<40)]
        #     df = df[(df.MACDh_12_26_9>0)]
        df = df[start-fit_length:start].copy()
        print('TRAIN LEN: ',len(df))
        # pass
    else:
        df = df[start:start+test_length].copy()
        print('TEST LEN: ',len(df))
        # quit()

    df_org = df.copy()
    # cols = ['MACD_12_26_9', 'MACDs_12_26_9', 'RSI_14']
    # cols = ['MACD_12_26_9', 'MACDs_12_26_9','BBL_signal','BBU_signal', 'RSI_14']
    # cols = ['RSI_14', 'BBL_signal', 'BBU_signal', 'MACDs_12_26_9', 'MACD_12_26_9', 'MACDh_12_26_9']
    # cols = ['RSI_14', 'RSI_shift1', 'RSI_shift2', 'RSI_shift3', 'RSI_shift4', 'RSI_shift5', 'RSI_shift6', 'RSI_shift7']
    # cols = ['RSI_14','RSI_shift1','RSI_shift2', 'RSI_shift3']
    # cols = ['close']
    # cols = ['RSI_shift6']
    # cols = ['close_shift2', 'close_shift3', 'close_shift4', 'close_shift5']
    cols = ['RSI_14', 'RSI_shift3']
    # columns_to_scale = ['MACD_12_26_9', 'MACDs_12_26_9','BBL_signal','BBU_signal']
    # df.RSI_14 /= 100
    df = df[cols].copy()

    pca = PCA(n_components=1)
    # principalComponents = pca.fit_transform(df)
    # print(principalComponents)


    # pca_transf = pca.fit_transform(df)
    # df = pd.DataFrame(data=pca_transf, columns=[f'pc_{i+1}' for i in range(pca_transf.shape[1])])


    
    # print(df), quit()

    # scale
    # df[columns_to_scale] = df[columns_to_scale].apply(zscore)
    # scaler = MinMaxScaler()
    # df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, df_org
    # return df.copy().to_numpy().reshape(-1,len(df.columns)), df_org


test_length = 90
fit_length = 90
start = random.randint(fit_length,len(df)-test_length)
Y_train, _ = process(df, start, 'train')
Y_test, df = process(df, start, 'test')


x = list(range(test_length))

contam = 0.1
clf = IsolationForest(contamination=contam, bootstrap=False, max_samples=0.99, n_estimators=200).fit(Y_test)
predictions = clf.predict(Y_test) == -1

# clf_valley = IsolationForest(contamination=contam, bootstrap=True, max_samples=0.8, n_estimators=300).fit(Y_train_valley)
# pred_valley = clf_valley.predict(Y_test) == -1


pred_valley, pred_peak = list(), list()
for idx, pred in enumerate(predictions):
    if pred and df.RSI_SMA_diff.iloc[idx]<0:
        pred_valley.append(True)
        pred_peak.append(False)
    elif pred and df.RSI_SMA_diff.iloc[idx]>0:
        pred_valley.append(False)
        pred_peak.append(True)
    else:
        pred_valley.append(False)
        pred_peak.append(False)

# print(df.close[pred])
# print(np.array(x)[pred])
# quit()



fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
ax1 = plt.subplot(4,1,1)
ax1.plot(x, df.close, color='gray')
# ax1.plot(x, df.EMA_12)
# ax1.plot(x, df.EMA_24)
ax1.plot(np.array(x)[pred_peak], df.close[pred_peak], '.', color='red')
ax1.plot(np.array(x)[pred_valley], df.close[pred_valley], '.', color='green')

ax2 = plt.subplot(4,1,2)
ax2.plot(x, df[['RSI_14', 'RSI_SMA']])

ax3 = plt.subplot(4,1,3)
ax3.plot(x, df[['RSI_SMA_diff']])

ax4 = plt.subplot(4,1,4)
ax4.plot(x, df[['MACDs_12_26_9']])

if "SSH_CONNECTION" in os.environ:
    print('Running from SSH -> fig saved')
    plt.savefig("latest_fig.png")
    quit()
else:
    plt.show()