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

import yfinance as yf

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split



class DataLoader:

    def __init__(self):
        self.run()

    def run(self):
        # Load files
        processed_folder = Path.cwd() / 'data' / 'processed'
        staged_folder = Path.cwd() / 'data' / 'staged'

        all_files = [x.stem for x in processed_folder.glob('*/')]
        tickers = list(set([i.split('_')[0] for i in all_files]))
        # tickers = ['DELL', 'TSLA']

        X = list()
        Y = list()
        rsi_diff_buy = {3:list(), 5:list(), 10:list()}
        rsi_diff_sell = {3:list(), 5:list(), 10:list()}

        # Concat each yearly df to a big df for each ticker
        for tick in tickers:
            print(f'Evaluating {tick} ...')
            files = list()
            for i in all_files:
                if tick == i.split('_')[0]:
                    files.append(i)
                    
            all_df = list()
            for file in files:
                all_df.append(pd.read_csv(processed_folder / f'{file}.csv', index_col='date'))
            
            df = pd.concat(all_df)
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
            df = df.asfreq(freq='1d', method='ffill')


            # remove nan rows
            df.dropna(axis=0, inplace=True)

            # Append yield curve
            # df = self._gen_yield_curve(df)

            # Change targets from [0,1,0] to 1
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


            # Save org df
            df_org = df.copy()


            # Choose selective columns
            cols = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14', 'target']
            df = df[cols]
            df.RSI_14 /= 100


            ## Split the data into buy, sell and hold sequences
            num_steps = 90
            columns_to_scale = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']

            df_hold = df[df.target == 0]
            df_buy = df[df.target == 1]
            df_sell = df[df.target == 2]


            # Buy seq
            buy_seq = list()
            for idx in range(len(df_buy)):
                end_date = df_buy.iloc[idx].name
                start_date = end_date-pd.Timedelta(days=num_steps-1)
                df_slice = df.loc[start_date:end_date].copy()
                if len(df_slice) < num_steps:
                    continue
                # if df_slice.RSI_14.iloc[-1] > 0.3:
                #     continue                
                df_slice.drop(['target'], axis=1, inplace=True)

                # Save rsi diff
                for rsi_idx in rsi_diff_buy.keys():
                    rsi_diff_buy[int(rsi_idx)].append( (df_slice.RSI_14.iloc[-1] - df_slice.RSI_14.iloc[-int(rsi_idx)-1])/rsi_idx )


                # print(df_slice)
                # print(rsi_diff_buy)
                # df_slice.plot(subplots=True)
                # plt.savefig("latest_fig2.png"), quit()

                # Zscore and scale
                df_slice[columns_to_scale] = df_slice[columns_to_scale].apply(zscore)
                scaler = MinMaxScaler()
                df_slice[columns_to_scale] = scaler.fit_transform(df_slice[columns_to_scale])
                # print(df_slice)
                # df_slice.plot(subplots=True)
                # plt.savefig("latest_fig.png")

                # random_num = random.randint(0,100)
                # if random_num>90:
                #     self._plotter(df_org.loc[start_date:end_date+pd.Timedelta(days=num_steps)])
                
                buy_seq.append(df_slice.to_numpy())
            buy_seq = np.array(buy_seq, dtype=object)


            # Sell seq
            sell_seq = list()
            for idx in range(len(df_sell)):
                end_date = df_sell.iloc[idx].name
                start_date = end_date-pd.Timedelta(days=num_steps-1)
                df_slice = df.loc[start_date:end_date].copy()
                if len(df_slice) < num_steps:
                    continue
                # if df_slice.RSI_14.iloc[-1] < 0.8:
                #     continue                
                df_slice.drop(['target'], axis=1, inplace=True)



                # Save rsi diff
                for rsi_idx in rsi_diff_sell.keys():
                    rsi_diff_sell[int(rsi_idx)].append( (df_slice.RSI_14.iloc[-1] - df_slice.RSI_14.iloc[-int(rsi_idx)-1])/rsi_idx )

                # Zscore and scale
                df_slice[columns_to_scale] = df_slice[columns_to_scale].apply(zscore)
                scaler = MinMaxScaler()
                df_slice[columns_to_scale] = scaler.fit_transform(df_slice[columns_to_scale])

                sell_seq.append(df_slice.to_numpy())
            sell_seq = np.array(sell_seq, dtype=object)


            # Determine the shortest sequence - to get equal amount of sell, buy and hold samples
            shortest_seq = min(len(sell_seq), len(buy_seq))

            # Sample indicies for the shortest sequence
            buy_seq_idx = np.random.choice(range(len(buy_seq)), size=shortest_seq, replace=False)
            sell_seq_idx = np.random.choice(range(len(sell_seq)), size=shortest_seq, replace=False)

            # Filter out the arrays for respective indicies
            buy_seq = buy_seq[buy_seq_idx]
            sell_seq = sell_seq[sell_seq_idx]

            # Hold seq
            hold_idx = np.random.randint(num_steps+1,len(df_hold), size=shortest_seq)
            hold_seq = list()
            for idx in hold_idx:
                end_date = df_hold.iloc[idx].name
                start_date = end_date-pd.Timedelta(days=num_steps-1)
                df_slice = df.loc[start_date:end_date].copy()
                if len(df_slice) < num_steps-1:
                    continue
                df_slice.drop(['target'], axis=1, inplace=True)

                # Zscore and scale
                df_slice[columns_to_scale] = df_slice[columns_to_scale].apply(zscore)
                scaler = MinMaxScaler()
                df_slice[columns_to_scale] = scaler.fit_transform(df_slice[columns_to_scale])

                hold_seq.append(df_slice.to_numpy())
            hold_seq = np.array(hold_seq, dtype=object)


            # Append to big X and Y
            for (buy,sell,hold) in zip(buy_seq, sell_seq, hold_seq):
                X.append(np.array(buy))
                X.append(np.array(sell))
                X.append(np.array(hold))
                Y.append(np.array([0,1,0]))
                Y.append(np.array([0,0,1]))
                Y.append(np.array([1,0,0]))


            print(f'{tick} Done')
            print('X shape:', np.array(X).shape)
            print('Y shape:', np.array(Y).shape)
            print('='*5)




        X = np.array(X)
        Y = np.array(Y)
        print('All tickers preprocessed.')
        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        print('='*5)


        # Save RSI diff
        rsi_diff_folder = Path.cwd() / 'data' / 'rsi_diff'
        with open(rsi_diff_folder / 'rsi_diff_sell.pkl', 'wb') as handle:
            pickle.dump(rsi_diff_sell, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(rsi_diff_folder / 'rsi_diff_buy.pkl', 'wb') as handle:
            pickle.dump(rsi_diff_buy, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save staged data
        with open(staged_folder / 'staged_x.npy', 'wb') as f:
            np.save(f, X)
        with open(staged_folder / 'staged_y.npy', 'wb') as f:
            np.save(f, Y)



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

        # Create a df to merge and fil for missing dates
        df_yield = pd.DataFrame(
            data={'13w':treas_bill_short.to_list(), '5y':treas_bill_5y.to_list() ,'10y':treas_bill_10y.to_list() , '30y':treas_bill_30y.to_list()},
            index=treas_bill_short.index
        )
        df_yield = df_yield.asfreq(freq='1d', method='ffill')

        # Calculate the yield curve
        yield_curve = list()
        for i in df_yield.index:
            arr = df_yield.loc[i].to_numpy()
            k, _ = np.polyfit([0.25, 5, 10, 30], arr, 1)
            yield_curve.append(k)
        df_yield['yield_curve'] = yield_curve

        df = df.join(df_yield.yield_curve)
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

