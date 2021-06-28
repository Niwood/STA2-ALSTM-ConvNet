import os
import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import random

import yfinance as yf

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf





class Live:


    def __init__(self, tick=None):

        # Parameters
        self.tick = tick
        self.model_name = '1624653692'
        self.num_steps = 90
        self.quantile_thr = 0.01 #for RSI threshold affirmation - lower means tighter
        self.today = datetime.today().date()
        self.ticker = yf.Ticker(self.tick)

        # Paths
        self.rsi_diff_folder = Path.cwd() / 'data' / 'rsi_diff'
        self.model_path = Path.cwd() / 'models' / self.model_name

        self._fetch_hist()


    def _load(self):
        ''' LOAD NET MODEL '''
        self.model = tf.keras.models.load_model(self.model_path)

        ''' LOAD RSI DIFF '''
        with open(self.rsi_diff_folder / 'rsi_diff_sell.pkl', 'rb') as handle:
            self.rsi_diff_sell = pickle.load(handle)
        with open(self.rsi_diff_folder / 'rsi_diff_buy.pkl', 'rb') as handle:
            self.rsi_diff_buy = pickle.load(handle)

        # Determine rsi_diff quantile threshold
        self.rsi_diff_buy_thr = dict()
        self.rsi_diff_sell_thr = dict()
        for rsi_idx in self.rsi_diff_buy.keys():
            self.rsi_diff_buy_thr[rsi_idx] = np.quantile(self.rsi_diff_buy[rsi_idx], self.quantile_thr, axis=0) #generally negative
            self.rsi_diff_sell_thr[rsi_idx] = np.quantile(self.rsi_diff_sell[rsi_idx], 1-self.quantile_thr, axis=0) #generally positive

        

    def _fetch_hist(self):
        ''' FETCH HIST DATA '''
        hist = self.ticker.history(start=self.today-timedelta(days=self.num_steps*2), end=self.today)
        self.df = hist.iloc[len(hist)-self.num_steps::]
        print(self.df)
        
        
    def _fetch_now(self):
        ''' FETCH LIVE DATA - appends to self.df '''
        self.now = self.ticker.history(start=self.today, end=self.today)



    def _isoforest(self, df):
        ''' Generate anomalie detection based in isolation forest '''
        df_org = df.copy()

        df['RSI_SMA'] = df.RSI_14.rolling(window=5).mean()
        df['RSI_SMA_diff'] = (df.RSI_14 - df.RSI_SMA)
        df.dropna(inplace=True)

        cols = ['RSI_14', 'RSI_SMA_diff']
        df = df[cols]
        
        # Isolation Ofrest prediction
        clf = IsolationForest(contamination=0.08, bootstrap=False, max_samples=0.99, n_estimators=200).fit(df)
        predictions = clf.predict(df) == -1

        # Insert anomalies
        df.insert(0, 'anomaly', predictions)
        df_org = df_org.join(df.anomaly)

        return df_org


    def predict(self, df):
        ''' Prediction algo '''
        pass



if __name__ == '__main__':
    live = Live(tick='AAPL')
    print('=== EOL: live.py ===')