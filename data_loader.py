import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import glob
from pathlib import Path
import random
import pickle
from tqdm import tqdm

import yfinance as yf

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

from scaler import FractionalDifferencing
from external_features import ExternalFeatures


class DataLoader:

    def __init__(self):

        # Folders
        self.processed_folder = Path.cwd() / 'data' / 'processed'
        self.staged_folder = Path.cwd() / 'data' / 'staged'
        self.external_features_folder = Path.cwd() / 'data' / 'external_features'

        # Get all ticks
        self.tickers = [x.stem for x in self.processed_folder.glob('*/')]
        random.shuffle(self.tickers)
        # self.tickers = ['NLS'] #fast
        # self.tickers = ['NXJ'] #slow
        # self.tickers = self.tickers[0:10]

        # Time steps to be fed into the network
        self.num_steps = 30

        # Columns that need z-score scaling
        self.columns_to_scale = [
            'close',
            'high',
            'low',
            'MACD_12_26_9',
            'MACDh_12_26_9',
            'MACDs_12_26_9',
            'BBU_signal',
            'BBL_signal',
            'volatility'
            ]

        # External features
        # extfeat = ExternalFeatures()
        # extfeat.update()
        self.df_ext_feat = pd.read_csv(self.external_features_folder / 'external_features.csv', index_col='Date')
        self.df_ext_feat.index = pd.to_datetime(self.df_ext_feat.index, format='%Y-%m-%d')

        # Track fatal runs
        self.fatal = list()
        self.successfull_runs = list()

        # Run
        self.run()



    def run(self):

        # Init list for X and Y and actions
        X = list()
        Y = list()
        buys, sells, holds = 0, 0, 0
        # rsi_diff_buy = {3:list(), 5:list(), 10:list()}
        # rsi_diff_sell = {3:list(), 5:list(), 10:list()}

        # Concat each yearly df to a big df for each ticker
        for tick_idx, tick in enumerate(self.tickers):
            print(f'Evaluating {tick} ... {tick_idx+1} out of {len(self.tickers)}')


            # Read the tick csv
            df = pd.read_csv(self.processed_folder / f'{tick}.csv', index_col='date')
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')


            # Drop some columns that are not needed
            df.drop(['Dividends', 'Stock Splits', 'volume', 'open'], axis=1, inplace=True)

            # Volatility
            df['volatility'] = df.close.rolling(window=2).std()

            # Return

            # # Yield curve
            # df = self._gen_yield_curve(df)

            # # Commodities
            # df = self._gen_oil(df)
            # df = self._gen_gold(df)

            # # Forex
            # df = self._gen_eurusd(df)

            # # Market indicies
            # df = self._gen_dji(df)
            # df = self._gen_snp(df)
            # df = self._gen_nasdaq(df)

            # Join external features
            df = df.join(self.df_ext_feat)

            # Remove nan and inf rows - MUST be after adding external features
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(axis=0, inplace=True)


            # Fractional differencing and scaling
            for col in self.columns_to_scale:

                # Select the column to scale
                serie_to_scale = df[col]

                # Init fracdiff and start iteration
                fracdiff = FractionalDifferencing(serie_to_scale, lag_cutoff_perc_thrsh=0.5)
                converged, result_data, p_value, correlation, lag_cutoff_perc, d_value = fracdiff.iterate(verbose=False)

                # If fractional differencing was not able to converge
                if not converged:
                    print('FATAL: Fractional differencing was not able to converge:')
                    print('column:', col)
                    print('p_value:', p_value)
                    print('correlation:', correlation)
                    print('lag_cutoff_perc:', lag_cutoff_perc)
                    print('d_value:', d_value)
                    break
                else:
                    # print('SUCCESS: Fractional differencing converged:')
                    # print('column:', col)
                    # print('p_value:', p_value)
                    # print('correlation:', correlation)
                    # print('lag_cutoff_perc:', lag_cutoff_perc)
                    # print('d_value:', d_value)
                    # print('='*10)
                    pass

                # MinMax scale the data
                scaled_data = fracdiff.scale_minmax(result_data)
                
                # Replace the original column with the scaled
                df.drop([col], axis=1, inplace=True)
                df = df.join(scaled_data)
                df.rename(columns={"scaled_data": col}, inplace=True)


            # Check if Fractional differencing was successfull
            if not converged:
                print(f'Skipping {tick} since fractional differencing was not able to converge.')
                self.fatal.append(tick)
                print('='*5)
                continue

            # Remove nan - MUST be after fractional differencing
            df.dropna(axis=0, inplace=True)

            # Scale RSI
            df.RSI_14 /= 100


            # Change targets from [0,1,0] to 1, etc.
            new_target = list()
            for tar in df.target:
                tar = tar[1:-1]
                tar = tar.split(',')
                tar = [int(i) for i in tar]

                if tar == [1,0,0]:
                    new_target.append(0)
                elif tar == [0,1,0]:
                    new_target.append(1)
                elif tar == [0,0,1]:
                    new_target.append(2)
            df.target = new_target



            ''' Split the data into buy, sell and hold sequences '''
            # Define which columns need to be scaled



            df_hold = df[df.target == 0]
            df_buy = df[df.target == 1]
            df_sell = df[df.target == 2]


            # Buy seq
            buy_seq = list()
            for idx in tqdm(range(len(df_buy)), desc='Buy sequence'):

                end_index = df.index.get_loc(df_buy.iloc[idx].name)

                # Continue if end_index is smaller then the time step
                if end_index < self.num_steps:
                    continue

                start_index = end_index - self.num_steps
                df_slice = df.iloc[start_index:end_index].copy()
                

                if len(df_slice) < self.num_steps:
                    continue 
                
                df_slice.drop(['target'], axis=1, inplace=True)

                # Zscore and scale
                # df_slice[self.columns_to_scale] = df_slice[self.columns_to_scale].apply(zscore)
                # scaler = MinMaxScaler()
                # df_slice[self.columns_to_scale] = scaler.fit_transform(df_slice[self.columns_to_scale])
                
                _df_slice = df_slice.copy()
                buy_seq.append(_df_slice.to_numpy())
            buy_seq = np.array(buy_seq, dtype=object)


            # Sell seq
            sell_seq = list()
            for idx in tqdm(range(len(df_sell)), desc='Sell sequence'):
                
                end_index = df.index.get_loc(df_sell.iloc[idx].name)

                # Continue if end_index is smaller then the time step
                if end_index < self.num_steps:
                    continue

                start_index = end_index - self.num_steps
                df_slice = df.iloc[start_index:end_index].copy()


                if len(df_slice) < self.num_steps:
                    continue          
                df_slice.drop(['target'], axis=1, inplace=True)


                # Save rsi diff
                # for rsi_idx in rsi_diff_sell.keys():
                #     rsi_diff_sell[int(rsi_idx)].append( (df_slice.RSI_14.iloc[-1] - df_slice.RSI_14.iloc[-int(rsi_idx)-1])/rsi_idx )

                # Anomaly detection - if the last timestep is not an anomaly, continue
                # df_slice = self._isoforest(df_slice.copy())
                # if not df_slice.iloc[-1].peak_anomaly and not df_slice.iloc[-1].valley_anomaly:
                #     pass
                    # continue

                # Zscore and scale
                # df_slice[self.columns_to_scale] = df_slice[self.columns_to_scale].apply(zscore)
                # scaler = MinMaxScaler()
                # df_slice[self.columns_to_scale] = scaler.fit_transform(df_slice[self.columns_to_scale])

                sell_seq.append(df_slice.to_numpy())
            sell_seq = np.array(sell_seq, dtype=object)

    
            # Determine the shortest sequence - to get equal amount of sell, buy and hold samples
            shortest_seq = min(len(sell_seq), len(buy_seq))
            if shortest_seq == 0:
                continue

            print('Shortest sequence:', shortest_seq)

            # Sample indicies for the shortest sequence
            buy_seq_idx = np.random.choice(range(len(buy_seq)), size=shortest_seq, replace=False)
            sell_seq_idx = np.random.choice(range(len(sell_seq)), size=shortest_seq, replace=False)

            # Filter out the arrays for respective indicies
            buy_seq = buy_seq[buy_seq_idx]
            sell_seq = sell_seq[sell_seq_idx]

            # Hold seq
            hold_idx = np.random.randint(self.num_steps+1,len(df_hold), size=shortest_seq)
            hold_seq = list()
            for idx in tqdm(hold_idx, desc='Hold sequence'):
                
                end_index = df.index.get_loc(df_hold.iloc[idx].name)

                # Continue if end_index is smaller then the time step
                if end_index < self.num_steps:
                    continue

                start_index = end_index - self.num_steps
                df_slice = df.iloc[start_index:end_index].copy()

                
                if len(df_slice) < self.num_steps-1:
                    continue
                df_slice.drop(['target'], axis=1, inplace=True)

                # Anomaly detection - if the last timestep is an anomaly, continue
                # df_slice = self._isoforest(df_slice.copy())
                # if df_slice.iloc[-1].anomaly:
                #     continue

                # Zscore and scale
                # df_slice[self.columns_to_scale] = df_slice[self.columns_to_scale].apply(zscore)
                # scaler = MinMaxScaler()
                # df_slice[self.columns_to_scale] = scaler.fit_transform(df_slice[self.columns_to_scale])

                hold_seq.append(df_slice.to_numpy())
            hold_seq = np.array(hold_seq, dtype=object)

            # Save counts
            buys += len(buy_seq)
            sells += len(sell_seq)
            holds += len(hold_seq)
            tot = buys+holds+sells


            # Append to big X and Y
            for (buy,sell,hold) in zip(buy_seq, sell_seq, hold_seq):
                X.append(np.array(buy))
                X.append(np.array(sell))
                X.append(np.array(hold))
                Y.append(np.array([0,1,0]))
                Y.append(np.array([0,0,1]))
                Y.append(np.array([1,0,0]))


            self.successfull_runs.append(tick)
            print(f'{tick} Done')
            print('X shape:', np.array(X).shape)
            print('Y shape:', np.array(Y).shape)
            print('Buys/Sells/Holds:', buys/tot, sells/tot, holds/tot)
            print(f'Fatal runs: {len(self.fatal)} | Successfull runs {len(self.successfull_runs)}')
            print('='*5)

            # Save partial step
            partial_saves = [200 ,500, 1000, 2000]
            if tick_idx in partial_saves:
                self.save(X, Y, tick_idx)


        # Save
        self.save(X,Y,'max')




    def save(self, X, Y, name):
        X = np.array(X)
        Y = np.array(Y)

        print('All tickers preprocessed.')
        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        print('='*5)

        # Save staged data
        with open(self.staged_folder / f'staged_x_{name}.npy', 'wb') as f:
            np.save(f, X)
        with open(self.staged_folder / f'staged_y_{name}.npy', 'wb') as f:
            np.save(f, Y)
        print(f'SAVED_{name}')



    def _isoforest(self, df):
        # Generate anomalie detection based in isolation forest algo
        # It will set a target to 0 if there is no anomly detected
        df_org = df.copy()

        df['RSI_SMA'] = df.RSI_14.rolling(window=7).mean()
        df['RSI_SMA_diff'] = (df.RSI_14 - df.RSI_SMA)
        df.dropna(inplace=True)

        cols = ['RSI_14', 'RSI_SMA_diff']
        df = df[cols]
        
        clf = IsolationForest(contamination=0.1, bootstrap=False, max_samples=0.99, n_estimators=200).fit(df)
        predictions = clf.predict(df) == -1


        df.insert(0, 'anomaly', predictions)
        df_org = df_org.join(df.anomaly)
        df_org.fillna(False, inplace=True)

        
        df_org.insert(0, 'peak_anomaly', (df_org.anomaly & (df_org.RSI_14.diff() > 0)))
        df_org.insert(0, 'valley_anomaly', (df_org.anomaly & (df_org.RSI_14.diff() < 0)))
        df_org.drop(['anomaly'], axis=1, inplace=True)
        # print(df_org.head(30))
        # print(df_org.tail(30))
        # quit()
        
        
        # # Change all targets to 0 if no anomaly is detected
        # df.loc[(df.anomaly==False) & (df.target>0), 'target'] = 0
        # anomaly_df = df[['anomaly', 'target']]

        # df_org = df_org.join(anomaly_df, lsuffix='old')
        # df_org.drop(['targetold'], axis=1, inplace=True)
        # df_org.dropna(axis=0, inplace=True)

        return df_org


    def _gen_yield_curve(self, df):

        # Get the ticker for each bond
        treas_bill_short_ticker = yf.Ticker("^IRX")
        treas_bill_5y_ticker = yf.Ticker("^FVX")
        treas_bill_10y_ticker = yf.Ticker("^TNX")
        treas_bill_30y_ticker = yf.Ticker("^TYX")

        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        treas_bill_short = treas_bill_short_ticker.history(start=str(start), end=str(end), interval='1d').Close
        treas_bill_5y = treas_bill_5y_ticker.history(start=str(start), end=str(end), interval='1d').Close
        treas_bill_10y = treas_bill_10y_ticker.history(start=str(start), end=str(end), interval='1d').Close
        treas_bill_30y = treas_bill_30y_ticker.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        treas_bill_short.rename("13w", inplace=True)
        treas_bill_5y.rename("5y", inplace=True)
        treas_bill_10y.rename("10y", inplace=True)
        treas_bill_30y.rename("30y", inplace=True)

        # Create a df to merge and fill for missing dates
        df_yield = pd.concat([treas_bill_short, treas_bill_5y, treas_bill_10y, treas_bill_30y], axis=1)
        df_yield = df_yield.fillna(method="ffill")
        
        # Make sure we have for all dates
        df_yield = df_yield.asfreq(freq='1d', method='ffill')

        # Calculate the yield curve
        yield_curve = list()
        for i in df_yield.index:
            arr = df_yield.loc[i].to_numpy()
            k, _ = np.polyfit([0.25, 5, 10, 30], arr, 1)
            yield_curve.append(k)
        df_yield['yield_curve'] = yield_curve

        # Check large/small values
        if max(df_yield.yield_curve.to_list()) > 1 or min(df_yield.yield_curve.to_list()) < -1:
            print(f'ERROR: Yield curve produced large/small values (max:{max(df_yield.yield_curve.to_list())} min:{min(df_yield.yield_curve.to_list())}). You have to scale them.')
            quit()

        # Join yield curve with df and return
        df = df.join(df_yield.yield_curve)
        return df


    def _gen_gold(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        gold_tick = yf.Ticker("GC=F")
        gold = gold_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        gold.rename("gold", inplace=True)

        # Join with df
        df = df.join(gold)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _gen_oil(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        oil_tick = yf.Ticker("CL=F")
        oil = oil_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        oil.rename("oil", inplace=True)

        # Join with df
        df = df.join(oil)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _gen_eurusd(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        eurusd_tick = yf.Ticker("EURUSD=X")
        eurusd = eurusd_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        eurusd.rename("eurusd", inplace=True)

        # Join with df
        df = df.join(eurusd)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _gen_dji(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        dji_tick = yf.Ticker("^DJI")
        dji = dji_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        dji.rename("dji", inplace=True)

        # Join with df
        df = df.join(dji)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _gen_snp(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        snp_tick = yf.Ticker("^GSPC")
        snp = snp_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        snp.rename("snp", inplace=True)

        # Join with df
        df = df.join(snp)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _gen_nasdaq(self, df):
        # Get history
        start = df.index[0].date()
        end = df.index[-1].date() + pd.Timedelta(days=1)

        # Get the ticker
        nasdaq_tick = yf.Ticker("^IXIC")
        nasdaq = nasdaq_tick.history(start=str(start), end=str(end), interval='1d').Close

        # Change series name
        nasdaq.rename("nasdaq", inplace=True)

        # Join with df
        df = df.join(nasdaq)

        # Fill na
        df = df.fillna(method="ffill")

        return df


    def _plotter(self, df):
        ''' PLOT EX - UNCOMMENT TO SHOW PLOT '''

        print(df)
        # print(df.index[df.target==1]), quit()
        # df = df.iloc[0:60]
        # cols = ['close','MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14']
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
        
        ax1 = plt.subplot(4,1,1)
        ax1.plot(df.index, df[['close']], linestyle='-', marker='')
        ax1.scatter(df.index[df.target==1],df.close[df.target==1], color='g')
        ax1.scatter(df.index[df.target==2],df.close[df.target==2], color='r')

        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.plot(df.index, df.MACDs_12_26_9, label='MACDs_12_26_9')
        ax2.plot(df.index, df.MACD_12_26_9, label='MACD_12_26_9')
        ax2.plot(df.index, df.MACDh_12_26_9, label='MACDh_12_26_9')
        # ax2.legend()

        ax3 = plt.subplot(4,1,3, sharex=ax1)
        ax3.plot(df.index, df[['RSI_14']])

        try:
            ax4 = plt.subplot(4,1,4, sharex=ax1)
            ax4.plot(df.index, df[['yield_curve']])
        except:
            pass

        # plt.savefig("latest_fig.png")
        plt.show()
        quit()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd

    t = DataLoader()
    

    print('=== EOL: data_loader.py ===')

