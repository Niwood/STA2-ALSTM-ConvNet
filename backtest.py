from os import X_OK
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

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import tensorflow as tf

from environment import StockTradingEnv



class Backtest:

    def __init__(self):

        # Parameters
        self.model_name = '1624218069'
        self.num_steps = 90
        self.SIM_MAX_STEPS = 800
        self.quantile_thr = 0.01 #for RSI threshold affirmation - lower means tighter

        # Paths
        self.validation_data_folder = Path.cwd() / 'data' / 'raw' / 'validation'
        self.rsi_diff_folder = Path.cwd() / 'data' / 'rsi_diff'
        self.model_path = Path.cwd() / 'models' / self.model_name

        self.load()
        self.eval()


    def load(self):
        ''' LOAD VALIDATION DATA '''
        # Ticker
        self.tick = random.choice([x.stem for x in self.validation_data_folder.glob('*.csv')])

        # Sample raw data and read as df
        df = pd.read_csv(self.validation_data_folder / f'{self.tick}.csv')
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # TI
        df.ta.macd(fast=12, slow=26, append=True)
        df.ta.rsi(append=True)
        df.dropna(inplace=True)

        # Reset and save index
        date_index = df.index.copy()
        close = df.close.copy()
        df = df.reset_index(drop=False)

        # Choose selective columns
        cols = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14']
        df = df[cols]
        df.RSI_14 /= 100

        self.df = df

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


        # bins = 80
        # plt.subplot(4,1,1)
        # plt.hist(rsi_diff_sell[1], bins=bins)

        # plt.subplot(4,1,2)
        # plt.hist(rsi_diff_sell[3], bins=bins)

        # plt.subplot(4,1,3)
        # plt.hist(rsi_diff_sell[5], bins=bins)

        # plt.subplot(4,1,4)
        # plt.hist(rsi_diff_sell[10], bins=bins)

        # plt.show()
        # quit()

        ''' LOAD ENV '''
        self.env = StockTradingEnv(close, self.num_steps)
        self.env.reset()
        self.env.max_steps = self.SIM_MAX_STEPS

        ''' LOAD MODEL '''
        
        self.model = tf.keras.models.load_model(self.model_path)


    def eval(self):
        # Trigger
        trigger = Trigger()

        # Eval lists
        self.stats = {
            'buy': list(),
            'buy_pred': list(),
            'sell': list(),
            'sell_pred': list(),
            'rewards': list(),
            'buy_n_hold': list(),
            'asset_amount': list(),
            'last_draw_down': list()
            }


        last_draw_down = 0
        for step in tqdm(range(self.SIM_MAX_STEPS)):

            # Reset trigger
            trigger.reset()

            # Step through the df and convert to numpy + Reshape (samples, steps, features)
            df_slice = self.df.loc[step:self.num_steps-1+step]
            df_slice_org = df_slice.copy()
            # print(df_slice_org), quit()

            # Zscore and scale
            columns_to_scale = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
            df_slice[columns_to_scale] = df_slice[columns_to_scale].apply(zscore)
            scaler = MinMaxScaler()
            df_slice[columns_to_scale] = scaler.fit_transform(df_slice[columns_to_scale])
            # print(df_slice)
            # print(df_slice.describe())
            # quit()

            # Convert to numpy and reshape
            X = df_slice.to_numpy()
            X = X.reshape((1 , *X.shape))

            # Network predict
            prediction = self.model.predict(X)[0]
            action = np.argmax(prediction)
            # action = random.choice([0,1,2])
            trigger.set(f'Model predicts: {action}', action)

            # Buy/sell threshold
            if trigger.action in (1,2):
                if max(prediction) < 0.90:
                    trigger.set('Model predicts: 0', 0)

            # RSI threshold affirmation - if predicted hold
            if trigger.action == 0:
                for rsi_idx in self.rsi_diff_buy.keys(): #any of the buy/sell keys are ok since they are the same
                    rsi_grad = (df_slice.RSI_14.iloc[-1] - df_slice.RSI_14.iloc[-int(rsi_idx)-1]) / rsi_idx

                    if rsi_grad < self.rsi_diff_buy_thr[rsi_idx] and self.env.shares_held==0:
                        trigger.set(f'RSI threshold affirmation: Below {self.quantile_thr*100}% {rsi_idx} day(s) threshold -> buy', 1, override=False)

                    elif rsi_grad > self.rsi_diff_sell_thr[rsi_idx] and self.env.shares_held>0:
                        trigger.set(f'RSI threshold affirmation: Above {(1-self.quantile_thr)*100}% {rsi_idx} day(s) threshold -> sell', 2, override=False)

            # RSI assertion
            # if trigger.action==1 and df_slice_org.RSI_14.iloc[-1]>0.3:
            #     trigger.set('RSI assertion: Failed on buy -> hold', 0)
            # elif trigger.action==2 and df_slice_org.RSI_14.iloc[-1]<0.7:
            #     trigger.set('RSI assertion: Failed on sell -> hold', 0)

            # MACD assertion
            if trigger.action==1 and df_slice_org.MACDh_12_26_9.iloc[-1]>0:
                trigger.set('MACD assertion: Failed on buy -> hold', 0)
            elif trigger.action==2 and df_slice_org.MACDh_12_26_9.iloc[-1]<0:
                trigger.set('MACD assertion: Failed on sell -> hold', 0)

            # Stop loss assertion - only if a buy action has been performed
            if self.env.buy_triggers>0 and self.env.shares_held>0:
                last_draw_down = (self.env.current_price / last_buy_price) - 1
                if self.env.shares_held>0 and last_draw_down<-0.1:
                    trigger.set('StopLoss assertion: Failed on any -> sell', 2)

            # If no shares but get sell -> hold, only for estatic reasons
            if self.env.shares_held==0 and trigger.action==2:
                trigger.set('No shares but get sell -> hold', 0)
                
            # If shares but get buy -> hold, only for estatic reasons
            if self.env.shares_held>0 and trigger.action==1:
                trigger.set('All shares are held -> hold', 0)


            ''' --- COMMITED TO ACTION FROM THIS POINT --- '''
            if trigger.action == 1:
                last_buy_price = self.env.current_price
            elif trigger.action == 2:
                last_draw_down = 0
                last_buy_price = self.env.current_price
            elif trigger.action == 0:
                if self.env.shares_held==0:
                    last_draw_down = 0

            
            if trigger.action in (1,2):
                print('---> ',f'step {step}:',trigger.desc)

            # Step env
            reward, done = self.env.step(trigger.action)

            # Save metrics
            if trigger.action == 1:
                self.stats['buy'].append(self.env.buy_n_hold)
                self.stats['sell'].append(np.nan)
                self.stats['buy_pred'].append((max(prediction)+5)**2)
                self.stats['sell_pred'].append(np.nan)
            elif trigger.action == 2:
                self.stats['buy'].append(np.nan)
                self.stats['sell'].append(self.env.buy_n_hold)
                self.stats['buy_pred'].append(np.nan)
                self.stats['sell_pred'].append((max(prediction)+5)**2)
            elif trigger.action == 0:
                self.stats['buy'].append(np.nan)
                self.stats['sell'].append(np.nan)
                self.stats['buy_pred'].append(np.nan)
                self.stats['sell_pred'].append(np.nan)

            self.stats['rewards'].append(reward)
            self.stats['asset_amount'].append(self.env.shares_held)
            self.stats['buy_n_hold'].append(self.env.buy_n_hold)
            self.stats['last_draw_down'].append(last_draw_down)

            # Break if done
            if done: break


    def plotter(self):
        x = list(range(len(self.stats['buy'])))

        plt.subplot(4,1,1)
        plt.plot(self.stats['rewards'])

        plt.subplot(4,1,2)
        plt.plot(self.stats['buy_n_hold'])
        plt.scatter(x, self.stats['buy'], s=self.stats['buy_pred'], c="g", alpha=0.5, marker='^',label="Buy")
        plt.scatter(x, self.stats['sell'], s=self.stats['sell_pred'], c="r", alpha=0.5, marker='v',label="Sell")

        plt.subplot(4,1,3)
        plt.plot(self.stats['asset_amount'])

        plt.subplot(4,1,4)
        plt.plot(self.stats['last_draw_down'])

        # plt.savefig("latest_fig.png")
        plt.show()



class Trigger:
    def __init__(self):
        self.desc = None
        self.override = False
        self.action = None

    def set(self, description, action, override=False):
        if not self.override:
            self.desc = description
            self.override = override
            self.action = action

    def reset(self):
        self.desc = None
        self.override = False


if __name__ == '__main__':
    back = Backtest()
    back.plotter()
    print('=== EOL: backtest.py ===')

