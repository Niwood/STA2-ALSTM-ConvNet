import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from pathlib import Path
import os
import math
from datetime import datetime

import numba
from numba import njit

from numpy import log
from sklearn.preprocessing import MinMaxScaler


class FractionalDifferencing:

    def __init__(self, data, lag_cutoff_perc_thrsh=0.3):
        ''' 
        data has to be a pd serie
        '''

        self.data = data

        # Parameters
        self.lag_cutoff_perc_thrsh = lag_cutoff_perc_thrsh
        self.correlation_thrsh = 0.8
        self.lag_cutoff_perc_increment = [0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]

        # Rename the series
        self.data.rename('input', inplace=True)

        # Log of data
        # self.data = log(self.data)
        
        # Specify the list of d-values to be tested
        self.d_list = list(np.arange(0,1,0.02))

        # Timers
        self.ts_diff_seconds = 0
        self.ts_diff_1_seconds = 0

    
    def iterate(self, verbose=False):

        # Flag for determining if we have successfully converged
        FLAG = False

        # Loop until correlation threshold is met - or lag_cutoff_perc is too big
        self.t0 = datetime.now()
        count = 0
        while True:

            # Specify lag cutoff start - this will increase until p-value and correlation thresholds are met
            lag_cutoff_perc = self.lag_cutoff_perc_increment[count]

            # Perform differencing
            result_data, p_value, correlation, d_value = self.differencing(self.data, lag_cutoff_perc)

            # Check if the correlation is within threshold
            if correlation < self.correlation_thrsh:
                if verbose:
                    print(f'FractionalDifferencing failed:')
                    print('p_value:', p_value)
                    print('correlation:', correlation)
                    print('d_value:', d_value)
                    print('lag_cutoff_perc:', lag_cutoff_perc)
                    print('Continues iteration ..')

            else:
                # SUCCESSFUL - we have met the correlation thrsh
                FLAG = True
                break

            # UNSUCCESSFUL - The lag_cutoff_perc is too high and we have not been able to converge
            if lag_cutoff_perc >= self.lag_cutoff_perc_thrsh:
                break

            count += 1
        self.total_seconds = (datetime.now()-self.t0).total_seconds()
        return FLAG, result_data, p_value, correlation, lag_cutoff_perc, d_value


    def differencing(self, data, lag_cutoff_perc):

        # Calc lag cutoff
        lag_cutoff = math.ceil(lag_cutoff_perc * len(data))

        # Convert the series to a frame
        data_as_frame = data.to_frame()
        
        # Loop through all d values and perform differencing
        for d in self.d_list:

            t0_ts_diff = datetime.now()
            weights = self._getWeights(d, lag_cutoff)
            diff_result = self._ts_differencing(data.to_numpy(), lag_cutoff, weights)
            differencing_result = pd.Series(data=diff_result, index=data.index[len(data) - len(diff_result)::])
            self.ts_diff_seconds += (datetime.now() - t0_ts_diff).total_seconds()

            differencing_result.rename(d, inplace=True)
            data_as_frame = data_as_frame.join(differencing_result)
            
            

        # Metrics
        return self._metrics(data_as_frame)



    def scale_minmax(self, data):
        ''' Min Max Scaler - input pd serie '''
        
        # Convert to frame
        data = data.to_frame()

        # Scale
        
        skscaler = MinMaxScaler()
        
        scaled_data = skscaler.fit_transform(data).reshape(1,-1)[0]

        # Insert in the dataframe - to preserve the date indexing
        data['scaled_data'] = scaled_data

        return data.scaled_data


    def _metrics(self, data_as_frame):
        
        # Loop through each differencing to calculate the metrics
        for i, col in enumerate(self.d_list):

            # Copy the input and differencing and drop nan
            differencing_result = data_as_frame[['input',col]].copy()
            differencing_result.dropna(inplace=True)

            # Calculate p-value from adf
            # adf = adfuller(differencing_result[col])
            # p_value = adf[1]

            dw = durbin_watson(differencing_result[col]) # 0 1 2 3 4
            p_value = dw
            # print(f'DW value: {p_value}')

            # Return if we have a d-value that is below p-value thrsh
            # if p_value < 0.05:
            if p_value >= 3.5 or p_value <= 0.5:
                correlation = differencing_result[col].corr(differencing_result["input"])
                return differencing_result[col], p_value, correlation, col
        
        # If p-value thrsh not met
        p_value, correlation = 2, 0
        return differencing_result[col], p_value, correlation, col


    def _getWeights(self, d,lags):
        # return the weights from the series expansion of the differencing operator
        # for real orders d and up to lags coefficients
        w=[1]
        for k in range(1,lags):
            w.append(-w[-1]*((d-k+1))/k)
        w=np.array(w).reshape(-1,1) 
        return w


    # @staticmethod
    # @njit
    # def shifter(arr, num, fill_value=0):
    #     # Method to perform shift in an array
    #     result = np.empty_like(arr)
    #     if num > 0:
    #         result[:num] = fill_value
    #         result[num:] = arr[:-num]
    #     elif num < 0:
    #         result[num:] = fill_value
    #         result[:num] = arr[-num:]
    #     else:
    #         result[:] = arr
    #     return result


    @staticmethod
    @njit()
    def _ts_differencing(series, lag_cutoff, weights):
        # return the time series resulting from (fractional) differencing
        # for real orders order up to lag_cutoff coefficients
        # weights = _getWeights(order, lag_cutoff)

        # t0_tsdiff_1 = datetime.now()

        res = np.empty_like(series, dtype=np.float_)
        # print(res)
        # quit()
        for k in range(lag_cutoff):

            # With pandas series - does not work with jit
            # res += weights[k]*series.shift(k).fillna(0)

            # Perform shift in an array
            series_shifted = np.empty_like(series, dtype=np.float_)
            if k > 0:
                series_shifted[:k] = 0
                series_shifted[k:] = series[:-k]
            elif k < 0:
                series_shifted[k:] = 0
                series_shifted[:k] = series[-k:]
            else:
                series_shifted[:] = series

            # With numpy - works with jit
            _res = np.multiply(weights[k], series_shifted)
            res = np.add(res, _res)
            
        # self.ts_diff_1_seconds += (datetime.now() - t0_tsdiff_1).total_seconds()

        return res[lag_cutoff:]

        # try:
        #     return res[lag_cutoff:]
        # except:
        #     print(series)
        #     print(res)
        #     print(lag_cutoff)
        #     quit()




if __name__ == '__main__':

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
    columns_to_scale = ['close']

    folder = Path.cwd() / 'data' / 'processed'
    tick = 'TACO'
    _data = pd.read_csv(folder / f'{tick}.csv', index_col='date')
    _data['volatility'] = _data.close.rolling(window=2).std()

    for col in columns_to_scale:
        data = _data[col]

        fracdiff = FractionalDifferencing(data)
        convergence_flag, result_data, p_value, correlation, lag_cutoff_perc, d_value = fracdiff.iterate(verbose=False)
        scaled_data = fracdiff.scale_minmax(result_data)


    # print('convergence_flag:', convergence_flag)
    # print('p_value:', p_value)
    # print('correlation:', correlation)
    # print('lag_cutoff_perc:', lag_cutoff_perc)
    # print('d_value:', d_value)
    # print('lag_cutoff_perc:', lag_cutoff_perc)

    print(f'Total seconds: {fracdiff.total_seconds}')
    print(f'ts_differencing seconds: {fracdiff.ts_diff_seconds}')
    print(f'ts_1_differencing seconds: {fracdiff.ts_diff_1_seconds}')
    

    print('Done')


    # scaled_data.hist(bins=50)
    # if "SSH_CONNECTION" in os.environ:
    #     plt.savefig("latest_fig.png")
    # else:
    #     plt.show()