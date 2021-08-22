import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from pathlib import Path
import os

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

    
    def iterate(self, verbose=False):

        # Flag for determining if we have successfully converged
        FLAG = False

        # Loop until correlation threshold is met - or lag_cutoff_perc is too big
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
        return FLAG, result_data, p_value, correlation, lag_cutoff_perc, d_value


    def differencing(self, data, lag_cutoff_perc):

        # Calc lag cutoff
        lag_cutoff = int(lag_cutoff_perc * len(data))

        # Convert the series to a frame
        data_as_frame = data.to_frame()
        
        # Loop through all d values and perform differencing
        for d in self.d_list:
            differencing_result = self._ts_differencing(data, d, lag_cutoff)
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
            adf = adfuller(differencing_result[col])
            p_value = adf[1]

            # Return if we have a d-value that is below p-value thrsh
            if p_value < 0.05:
                correlation = differencing_result[col].corr(differencing_result["input"])
                return differencing_result[col], p_value, correlation, col
        
        # If p-value thrsh not met
        p_value, correlation = 1, 0
        return differencing_result[col], p_value, correlation, col


    def _getWeights(self, d,lags):
        # return the weights from the series expansion of the differencing operator
        # for real orders d and up to lags coefficients
        w=[1]
        for k in range(1,lags):
            w.append(-w[-1]*((d-k+1))/k)
        w=np.array(w).reshape(-1,1) 
        return w


    def _ts_differencing(self, series, order, lag_cutoff):
        # return the time series resulting from (fractional) differencing
        # for real orders order up to lag_cutoff coefficients
        
        weights = self._getWeights(order, lag_cutoff)
        res = 0
        for k in range(lag_cutoff):
            res += weights[k]*series.shift(k).fillna(0)
        return res[lag_cutoff:] 



if __name__ == '__main__':

    folder = Path.cwd() / 'data' / 'processed'
    tick = 'AAPL'
    _data = pd.read_csv(folder / f'{tick}.csv', index_col='date')
    data = _data.close

    fracdiff = FractionalDifferencing(data)
    convergence_flag, result_data, p_value, correlation, lag_cutoff_perc, d_value = fracdiff.iterate(verbose=True)
    scaled_data = fracdiff.scale_minmax(result_data)


    print('convergence_flag:', convergence_flag)
    print('p_value:', p_value)
    print('correlation:', correlation)
    print('lag_cutoff_perc:', lag_cutoff_perc)
    print('d_value:', d_value)
    print('lag_cutoff_perc:', lag_cutoff_perc)


    scaled_data.hist(bins=50)
    if "SSH_CONNECTION" in os.environ:
        plt.savefig("latest_fig.png")
    else:
        plt.show()