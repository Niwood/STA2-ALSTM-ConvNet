import os
import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib import ticker
import numpy as np
from datetime import datetime, timedelta
import pytz
import glob
from pathlib import Path
from tqdm import tqdm
import pickle
import random
import time

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

import tensorflow as tf
import tflite_runtime.interpreter as tflite

from environment import StockTradingEnv
from utils.feature_engineer import FeatureEngineer


class Backtest:

    def __init__(self):
        
        # Parameters
        self.DEBUG = False
        self.model_name = '1630867357_lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
        self.num_steps = 30
        self.SIM_MAX_STEPS = 500
        
        # Paths
        # self.validation_data_folder = Path.cwd() / 'data' / 'raw' / 'validation'
        self.validation_data_folder = Path.cwd() / 'data' / 'raw'
        self.model_path = Path.cwd() / 'models' / self.model_name 
        # self.model_path = Path.cwd() / 'tflite_models' / f'{self.model_name}.tflite'

        # Feature Engineer
        self.feature_engineer = FeatureEngineer()

        # Load saved model
        if not self.DEBUG:
            self.load_model()


    def run(self):
        eval_ok = False
        tick_ok = False
        try:
            while not (eval_ok and tick_ok):
                try:
                    tick_ok = self.load_tick()
                except:
                    tick_ok = False
                if tick_ok:
                    try:
                        eval_ok = self.eval()
                    except Exception as e:
                        print(f'Evaluation NOK: {e}')
                        eval_ok = False
        except KeyboardInterrupt:
            print('Interrupted')

    def load_tick(self):
        ''' LOAD MODEL AND VALIDATION DATA + FEATURE ENGINEERING '''

        # Sample a tick
        self.tick = random.choice([x.stem for x in self.validation_data_folder.glob('*.pkl')])

        # Read as df
        df = pd.read_pickle(self.validation_data_folder / f'{self.tick}.pkl')
        
        if len(df) == 0:
            return False

        # df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)#.tz_convert('UTC')
        
        # Feature engineer
        df = self.feature_engineer.first_process(df)
        if not self.DEBUG:
            df, converged = self.feature_engineer.second_process(df)
        else:
            converged = True

        if not converged:
            print(f'{self.tick} did not converge, will resample tick')
            return False

        # Reset and save index
        self.date_index = df.index.copy().to_list()
        close = df.close.copy()
        df = df.reset_index(drop=False)
        # print(df), quit()

        # Selective columns for the model
        if not self.DEBUG:
            df = self.feature_engineer.select_columns(df)
        else:
            df = df[['close', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 
                'MACDs_12_26_9', 'BBU_signal',
                'BBL_signal'
                ]]

        self.df = df
        return True


    def load_model(self):
        ''' LOAD MODEL '''
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model.summary()), quit()
        # self.model = tflite.Interpreter(model_path=str(self.model_path))


    def eval(self):
        
        if len(self.df) - (self.SIM_MAX_STEPS + self.num_steps*2) < 0:
            print(f'{self.tick} was too short, sampling a new tick..')
            return False


        # Sample df to perform backtest on, len(df_backtest)=SIM_MAX_STEPS
        self.start_step = random.randint(0,len(self.df) - (self.SIM_MAX_STEPS + self.num_steps))

        self.df_backtest = self.df.iloc[self.start_step:self.start_step+self.SIM_MAX_STEPS]


        # Save the start and end date
        # self.start_date = self.df_backtest.index[0]
        self.start_date = self.date_index[self.start_step]
        self.end_date = self.df_backtest.index[-1]
        self.date_axis = list()
        
        # Load env
        self.env = StockTradingEnv(self.df_backtest, self.num_steps)
        self.env.reset()
        self.env.max_steps = self.SIM_MAX_STEPS

        # Trigger
        trigger = Trigger()

        # Gain
        self.gain = Gain()

        # Eval lists
        self.stats = {
            'buy': list(),
            'buy_pred': list(),
            'sell': list(),
            'sell_pred': list(),
            'rewards': list(),
            'buy_n_hold': list(),
            'asset_amount': list(),
            'last_draw_down': list(),
            'close_rsi': list(),
            'close_macd': list(),
            'reward_sma': list(),
            'net_worth': list(),
            'stock_price': list()
            }
        # self.v_anomalies = list()
        # self.p_anomalies = list()

        # anomaly_stats = {'Hold':0, 'Buy':0, 'Sell':0, 'Total':0}
        for step, df_index in enumerate(tqdm(self.df_backtest.index)):
            

            current_date = self.date_index[self.start_step+step]
            self.date_axis.append(current_date)

            # Reset trigger
            trigger.reset()

            # Update gain
            self.gain.update(self.df_backtest.close.iloc[step], current_date)

            # Step through the df and convert to numpy + Reshape (samples, steps, features)
            # slice = (df_index - self.num_steps + 1, df_index)
            df_slice = self.df.loc[df_index - self.num_steps + 1 : df_index]
            
            # self.current_date = self.df_backtest.date[step]
            df_slice_org = df_slice.copy()
            self.stats['close_rsi'].append(df_slice.RSI_14.iloc[-1])
            self.stats['close_macd'].append(df_slice.MACDs_12_26_9.iloc[-1])

            
            # Anomaly detection
            # df_slice = self._isoforest(df_slice.copy())


            # Convert to numpy and reshape
            X = df_slice.to_numpy()
            X = X.reshape((1 , *X.shape))

            # Edit format to float32
            X = np.asarray(X).astype('float32')

            # Network predict

            
            # if df_slice.peak_anomaly.iloc[-1] or df_slice.valley_anomaly.iloc[-1]:
            #     prediction = self.model.predict(X)[0]
            #     action = np.argmax(prediction)
            #     trigger.set(f'Model predicts: {action}', action, override=False)
            # else:
            #     action = 0
            #     trigger.set(f'Model predicts: {action}', action, override=False)

            if not self.DEBUG:
                prediction = self.model.predict(X)[0]
            else:
                rand = random.random()
                if rand > 0.1:
                    prediction = [0.8, 0.1, 0.1]
                else:
                    prediction = random.choice(
                        [[0.1, 0.8, 0.1],[0.1, 0.1, 0.8]]
                        )
            action = np.argmax(prediction)
            trigger.set(f'Model predicts: {action} with probabilty {max(prediction)}', action, override=False)

            # Model certainty threshold
            if trigger.action == 1:
                if max(prediction) < 0.80:
                    trigger.set(f'Below model certainty threshold ({str(round(max(prediction),2))}): {action} -> 0', 0)


            # RSI assertion
            # if trigger.action==1 and df_slice_org.RSI_14.iloc[-1]>0.5:
            #     trigger.set('RSI assertion: Failed on buy -> hold', 0)
            # elif trigger.action==2 and df_slice_org.RSI_14.iloc[-1]<0.5:
            #     trigger.set('RSI assertion: Failed on sell -> hold', 0)

            # MACD assertion
            # if trigger.action==1 and df_slice_org.MACDh_12_26_9.iloc[-1]>0:
            #     trigger.set('MACD assertion: Failed on buy -> hold', 0)
            # elif trigger.action==2 and df_slice_org.MACDh_12_26_9.iloc[-1]<0:
            #     trigger.set('MACD assertion: Failed on sell -> hold', 0)

            # Stop loss assertion - only if a buy action has been performed
            # if self.env.buy_triggers>0 and self.env.shares_held>0:
            #     if self.env.shares_held>0 and self.gain.gain>=0.05:
            #         trigger.set(f'Inverted StopLoss assertion ({str(round(self.gain.gain,3))}): Failed on any -> sell', 2)
            #     elif self.env.shares_held>0 and self.gain.gain<-0.02:
            #         trigger.set(f'StopLoss assertion ({str(round(self.gain.gain,3))}): Failed on any -> sell', 2)

            # If no shares but get sell -> hold, only for estatic reasons
            if self.env.shares_held==0 and trigger.action==2:
                trigger.set('No shares but get sell -> hold', 0, override=False)
                
            # If shares but get buy -> hold, only for estatic reasons
            if self.env.shares_held>0 and trigger.action==1:
                trigger.set('All shares are held -> hold', 0, override=False)


            ''' --- COMMITED TO ACTION FROM THIS POINT --- '''
            if trigger.action == 1:
                self.gain.buy(self.df_backtest.close.iloc[step], current_date)
                trigger.set_id()

            elif trigger.action == 2:
                self.gain.sell()

            # elif trigger.action == 0:
            #     self.gain.hold(self.env.current_price)


            # if df_slice.peak_anomaly.iloc[-1] or df_slice.valley_anomaly.iloc[-1]:
            #     if df_slice.valley_anomaly.iloc[-1]:
            #         self.v_anomalies.append(step)
            #         self.p_anomalies.append(np.nan)
            #     elif df_slice.peak_anomaly.iloc[-1]:
            #         self.p_anomalies.append(step)
            #         self.v_anomalies.append(np.nan)

            #     anomaly_stats['Total'] += 1
            #     if trigger.action == 0:
            #         anomaly_stats['Hold'] += 1
            #     elif trigger.action == 1:
            #         anomaly_stats['Buy'] += 1
            #     elif trigger.action == 2:
            #         anomaly_stats['Sell'] += 1
            # else:
            #     self.v_anomalies.append(np.nan)
            #     self.p_anomalies.append(np.nan)

            if trigger.action in (1,2):
                print('---> ',f'step {step}:',trigger.desc, 'Trigger ID:',trigger.id)

            # Step env
            reward, done = self.env.step(trigger.action)

            # Save metrics
            if trigger.action == 1:
                self.stats['buy'].append(self.env.buy_n_hold*100)
                self.stats['sell'].append(np.nan)
                self.stats['buy_pred'].append((max(prediction)+5)**2)
                self.stats['sell_pred'].append(np.nan)
            elif trigger.action == 2:
                self.stats['buy'].append(np.nan)
                self.stats['sell'].append(self.env.buy_n_hold*100)
                self.stats['buy_pred'].append(np.nan)
                self.stats['sell_pred'].append((max(prediction)+5)**2)
            elif trigger.action == 0:
                self.stats['buy'].append(np.nan)
                self.stats['sell'].append(np.nan)
                self.stats['buy_pred'].append(np.nan)
                self.stats['sell_pred'].append(np.nan)

            self.stats['rewards'].append(reward*100)
            self.stats['asset_amount'].append(self.env.shares_held)
            self.stats['buy_n_hold'].append(self.env.buy_n_hold*100)
            self.stats['last_draw_down'].append(self.gain.gain)
            self.stats['reward_sma'].append(pd.Series(self.stats['rewards']).rolling(window=7).mean().iloc[-1])
            self.stats['net_worth'].append(self.env.net_worth)
            self.stats['stock_price'].append(self.env.current_price)

            # Break if done
            if done: break
        

        print(f'{self.tick} -> from {self.start_date} to {self.end_date}')

        return True


    def plotter(self):
        x = list(range(len(self.stats['buy'])))
        fig, (ax1, ax2, ax4) = plt.subplots(3,1)
        

        ''' Portfolio value / return '''
        ax1 = plt.subplot(3,1,1)
        plt.title(f'Tick: {self.tick}')
        ax1_2 = ax1.twinx()
        # Plots
        ax1.plot(self.date_axis, self.stats['net_worth'])
        ax1_2.plot(self.date_axis, self.stats['rewards'], label='Return')
        # Labels
        ax1.set_ylabel('Portfolio Value')
        ax1_2.set_ylabel('Return')
        # Format axis
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax1_2.yaxis.set_major_formatter(mtick.PercentFormatter())
        

        ''' Stock price / change '''
        ax2 = plt.subplot(3,1,2, sharex=ax1)
        ax2_2 = ax2.twinx()
        # Plot
        ax2.plot(self.date_axis, self.stats['stock_price'])
        ax2_2.plot(self.date_axis, self.stats['buy_n_hold'], label='Change')
        ax2_2.scatter(self.date_axis, self.stats['buy'], s=80, color="green", alpha=0.8, marker='^', label="Buy")
        ax2_2.scatter(self.date_axis, self.stats['sell'], s=80, color="red", alpha=0.8, marker='v', label="Sell")
 
        
        # ax2_2.plot(self.close_change, color='r', linestyle='--')
        # for anom_x in self.v_anomalies:
        #     ax2.axvline(anom_x, 0, 1, color='lime', linestyle='dashed', alpha=0.2)
        # for anom_x in self.p_anomalies:
        #     ax2.axvline(anom_x, 0, 1, color='magenta', linestyle='dashed', alpha=0.2)            
        
        # Labels
        ax2.set_ylabel('Stock Price')
        ax2_2.set_ylabel('Change')
        # Format axis
        # ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax2_2.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Buy/Sell triggers
        

        ''' RSI '''
        # ax3 = plt.subplot(4,1,3, sharex=ax1)
        # plt.plot(self.reward_rsi)
        # plt.plot(self.stats['asset_amount'])
        # ax3.plot(self.stats['close_rsi'])
        # for anom_x in self.v_anomalies:
        #     ax3.axvline(anom_x, 0, 1, color='lime', linestyle='dashed', alpha=0.2)
        # for anom_x in self.p_anomalies:
        #     ax3.axvline(anom_x, 0, 1, color='magenta', linestyle='dashed', alpha=0.2)  


        ''' Position gain '''
        ax4 = plt.subplot(3,1,3, sharex=ax1)
        for x,y in zip(self.gain.gains_master_xaxis, self.gain.gains_master):
            if y[-1] >= 0:
                _color = 'g'
            elif y[-1] < 0:
                _color = 'r'


            ax4.fill_between(x, 0, 1, where= np.array(y)<np.inf,
                color=_color, alpha=0.2, transform=ax4.get_xaxis_transform(), linestyle='-', edgecolor=_color)
            ax2_2.fill_between(x, 0, 1, where= np.array(y)<np.inf,
                color=_color, alpha=0.2, transform=ax2_2.get_xaxis_transform(), linestyle='-', edgecolor=_color)
            ax1_2.fill_between(x, 0, 1, where= np.array(y)<np.inf,
                color=_color, alpha=0.2, transform=ax1_2.get_xaxis_transform(), linestyle='-', edgecolor=_color)
            ax4.plot(x, y, color=_color)


        # Labels
        ax4.set_ylabel('Individual position return')
        # Format axis
        ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
        

        
        ''' Figure settings '''
        plt.gcf().set_size_inches(15, 10, forward=True)
        
        fig.autofmt_xdate()
        ax1.grid()
        ax2.grid()
        ax4.grid()
        # multi = MultiCursor(fig.canvas, (ax1, ax2, ax4), color='r', lw=1)

        if "SSH_CONNECTION" in os.environ:
            print('Running from SSH -> fig saved')
            plt.savefig("latest_fig.png")
            quit()
        else:
            plt.savefig(f'backtest_results/{self.tick}.pdf')  
            # plt.show()



class Trigger:
    def __init__(self):
        self.id = 0.0
        self.desc = None
        self.override = False
        self.action = None

    def set_id(self):
        self.id = time.time()

    def set(self, description, action, override=False):
        if not self.override:
            self.desc = description
            self.override = override
            self.action = action

    def reset(self):
        self.desc = None
        self.override = False



class Gain:

    def __init__(self):
        self.buy_price = 0
        self.gain = 0
        self.gains = list()
        self.gains_master = list()
        self.gains_master_xaxis = list()
        self.previous_step_was_buy = False


    def update(self, price, date):

        if self.previous_step_was_buy:
            self.gains[-1] = 0
            self.previous_step_was_buy = False

        if self.buy_price > 0:
            self.gain = (price / self.buy_price) - 1
            self.position_gain.append(((price / self.buy_price) - 1)*100)
            self.position_gain_xaxis.append(date)
        else:
            self.gain = np.nan
        self.gains.append(self.gain*100)


    def buy(self, buy_price, date):
        self.buy_price = buy_price
        self.previous_step_was_buy = True
        self.position_gain = [0]
        self.position_gain_xaxis = [date]
        # print('----> FROM GAIN:', self.buy_price)


    def sell(self):
        self.gains_master.append(self.position_gain)
        self.gains_master_xaxis.append(self.position_gain_xaxis)
        self.buy_price = 0
        self.gain = 0





if __name__ == '__main__':

    back = Backtest()
    
    back.run()
    back.plotter()
    print('=== EOL: backtest.py ===')

