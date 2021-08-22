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

from scaler import FractionalDifferencing





class ExternalFeatures:

    def __init__(self):

        # Folders
        self.external_features_folder = Path.cwd() / 'data' / 'external_features'


    def update(self):
        # Specify methods to generate the external features
        yield_curve = self._gen_yield_curve()
        gold = self._gen_gold()
        oil = self._gen_oil()
        eurusd = self._gen_eurusd()
        dji = self._gen_dji()
        snp = self._gen_snp()
        nasdaq = self._gen_nasdaq()

        # Merge all series into a df
        df = yield_curve.to_frame()
        df = df.join(gold)
        df = df.join(oil)
        df = df.join(eurusd)
        df = df.join(dji)
        df = df.join(snp)
        df = df.join(nasdaq)

        # Drop nan
        df.dropna(axis=0, inplace=True)

        # Save
        df.to_csv(self.external_features_folder / 'external_features.csv' ,index=True)
        print('External features have been updated and saved.')


    def _gen_yield_curve(self):

        # Get the ticker for each bond
        treas_bill_short_ticker = yf.Ticker("^IRX")
        treas_bill_5y_ticker = yf.Ticker("^FVX")
        treas_bill_10y_ticker = yf.Ticker("^TNX")
        treas_bill_30y_ticker = yf.Ticker("^TYX")

        
        treas_bill_short = treas_bill_short_ticker.history(period="max", interval='1d').Close
        treas_bill_5y = treas_bill_5y_ticker.history(period="max", interval='1d').Close
        treas_bill_10y = treas_bill_10y_ticker.history(period="max", interval='1d').Close
        treas_bill_30y = treas_bill_30y_ticker.history(period="max", interval='1d').Close

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

        # Drop nan
        df_yield.dropna(axis=0, inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(df_yield.yield_curve)

        return scaled_data


    def _gen_gold(self):

        # Get the ticker
        gold_tick = yf.Ticker("GC=F")
        gold = gold_tick.history(period="max", interval='1d').Close

        # Change series name
        gold.rename("gold", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(gold)

        return scaled_data


    def _gen_oil(self):

        # Get the ticker
        oil_tick = yf.Ticker("CL=F")
        oil = oil_tick.history(period="max", interval='1d').Close

        # Change series name
        oil.rename("oil", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(oil)

        return scaled_data


    def _gen_eurusd(self):

        # Get the ticker
        eurusd_tick = yf.Ticker("EURUSD=X")
        eurusd = eurusd_tick.history(period='max', interval='1d').Close

        # Change series name
        eurusd.rename("eurusd", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(eurusd)

        return scaled_data


    def _gen_dji(self):

        # Get the ticker
        dji_tick = yf.Ticker("^DJI")
        dji = dji_tick.history(period='max', interval='1d').Close

        # Change series name
        dji.rename("dji", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(dji)

        return scaled_data


    def _gen_snp(self):

        # Get the ticker
        snp_tick = yf.Ticker("^GSPC")
        snp = snp_tick.history(period='max', interval='1d').Close

        # Change series name
        snp.rename("snp", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(snp)

        return scaled_data


    def _gen_nasdaq(self):

        # Get the ticker
        nasdaq_tick = yf.Ticker("^IXIC")
        nasdaq = nasdaq_tick.history(period='max', interval='1d').Close

        # Change series name
        nasdaq.rename("nasdaq", inplace=True)

        # Fractional differencing and scale
        scaled_data = self._process(nasdaq)

        return scaled_data


    def _process(self, data):
        
        feature_name = data.name
        print(f'Calculating fractional differencing on external feature: {feature_name}')

        # Init fracdiff and start iteration
        fracdiff = FractionalDifferencing(data, lag_cutoff_perc_thrsh=0.5)
        converged, result_data, p_value, correlation, lag_cutoff_perc, d_value = fracdiff.iterate(verbose=False)

        # If fractional differencing was not able to converge
        if not converged:
            print('FATAL: Fractional differencing for external feauture was not able to converge:')
            print('p_value:', p_value)
            print('correlation:', correlation)
            print('lag_cutoff_perc:', lag_cutoff_perc)
            print('d_value:', d_value)
            quit()
        else:
            print('SUCCESS: Fractional differencing converged:')
            print('p_value:', p_value)
            print('correlation:', correlation)
            print('lag_cutoff_perc:', lag_cutoff_perc)
            print('d_value:', d_value)
            print('='*10)

        # MinMax scale the data
        scaled_result_data = fracdiff.scale_minmax(result_data)
        scaled_result_data.rename(feature_name, inplace=True)

        return scaled_result_data



if __name__ == '__main__':
    exfeat = ExternalFeatures()
    exfeat.update()