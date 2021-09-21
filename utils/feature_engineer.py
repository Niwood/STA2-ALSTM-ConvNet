import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
        
from utils.decorater import check_columns
from utils.external_features import ExternalFeatures
from utils.scaler import FractionalDifferencing


class FeatureEngineer:
    '''
    FeatureEngineer will perform feature engineering in two steps.
    '''


    def __init__(self):

        # Load external features
        external_features_folder = Path.cwd() / 'data' / 'external_features'
        self.df_ext_feat = pd.read_csv(external_features_folder / 'external_features.csv', index_col='Date')
        self.df_ext_feat.index = pd.to_datetime(self.df_ext_feat.index, format='%Y-%m-%d')
        

    def first_process(self, df):
        ''' 
        The first process sets the date and technical indicators
        Used by label processer and backtesting
        '''

        # Add day/month of year to df
        df.insert(1, "day_of_year", (df.index.day_of_year-1)/366, True)
        df.insert(1, "month_of_year", (df.index.month-1)/12, True)

        try:
            # MACD and RSI
            df.ta.macd(fast=12, slow=26, append=True)
            df.ta.rsi(append=True)
            
            # BBAND - upper/lower signal, percentage of how close the hlc is to upper/lower bband
            bband_length = 30
            bband = df.copy().ta.bbands(length=bband_length)
            bband['hlc'] = df.copy().ta.hlc3()

            bbu_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBU_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])
            bbl_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBL_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])

            df['BBU_signal'] = bbu_signal
            df['BBL_signal'] = bbl_signal

        except Exception as e:
            print(df)
            assert False, e

        # Drop NA rows
        df.dropna(inplace=True)

        return df


    def second_process(self, df):
        '''
        The second process adds extra features and scales
        Used by data loader and backtesting
        '''

        # Columns to scale
        columns_to_scale = [
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

        # Volatility
        df['volatility'] = df.close.rolling(window=2).std()

        # Join external features: if the index columns is a range index -> select the date column as key for the join
        if type(df.index) == pd.core.indexes.range.RangeIndex:
            df = pd.merge(
                df, #left
                self.df_ext_feat, #right
                how="inner",
                left_on='date',
                right_index=True)
        else:
            df = df.join(self.df_ext_feat)


        # Remove nan and inf rows - MUST be after adding external features
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, inplace=True)

        # Fractional differencing and scaling
        for col in columns_to_scale:

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


        # Remove nan - MUST be after fractional differencing
        df.dropna(axis=0, inplace=True)

        # Scale RSI
        df.RSI_14 /= 100

        return df, converged

    def select_columns(self, df):
        columns = [
            'month_of_year', 'day_of_year', 'RSI_14', 'yield_curve',
            'gold', 'oil', 'eurusd', 'dji', 'snp', 'nasdaq', 'close', 'high', 'low',
            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'BBU_signal',
            'BBL_signal', 'volatility'
            ]
        return df[columns]


    def update_external_features(self):
        extfeat = ExternalFeatures()
        extfeat.update()


if __name__ == '__main__':
    ft = FeatureEngineer()