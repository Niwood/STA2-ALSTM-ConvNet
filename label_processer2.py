import os
import sys
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import  datetime
import glob
from pathlib import Path
from scipy.signal import find_peaks, find_peaks_cwt
import random
import statsmodels.api as sm

from sklearn.ensemble import IsolationForest

from datetime import datetime





class LP:

    def __init__(self):

        # PARAMETERS
        self.DEBUG = False
        self.SHUFFLE, self.SHUFFLE_SEED = True, 543
        # small: 543

        # Handle randomness
        if self.SHUFFLE:
            self.SHUFFLE_SEED = random.randrange(sys.maxsize)
        if self.DEBUG:
            print('SEED NUMBER:', self.SHUFFLE_SEED)

        # Folders
        self.raw_folder = Path.cwd() / 'data' / 'raw'
        self.processed_folder = Path.cwd() / 'data' / 'processed'

        # Time steps to process (year, month)
        self.time_step = 'year'

        # Ticks already processed
        processed_ticks = set([x.stem for x in self.processed_folder.glob('*/')])

        # Get all ticks in raw folder
        all_ticks_in_raw = set([x.stem for x in self.raw_folder.glob('*.pkl')])

        # All remaining ticks that has not been processed and shuffle
        self.all_ticks = list(all_ticks_in_raw.difference(processed_ticks))
        random.Random(self.SHUFFLE_SEED).shuffle(self.all_ticks)
            
        # Run
        while True:
            done = self.run()
            if done:
                print('All ticks done.')
                break

    def save(self, frame, tick, date_buy, date_sell):

        # Insert target column all with hold action
        frame.insert(1, "target", [[1,0,0]]*len(frame))

        # Loop through all peaks and valleys and change target
        for x in date_buy:
            frame.at[x, 'target'] = [0,1,0]
        for x in date_sell:
            frame.at[x, 'target'] = [0,0,1]

        # Save
        frame.to_csv(self.processed_folder / f'{tick}.csv' ,index=True)


    def run(self):
        ''' Return True if done else False '''

        # Try to pick next tick
        try:
            tick = self.all_ticks[0]
            print('TICKS REM:', len(self.all_ticks))
        except:
            return True
        
        # Load data for the tick
        df = pd.read_pickle(self.raw_folder / f'{tick}.pkl')
        df.index = pd.to_datetime(df.index)

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
            print(f'TECHINCAL INDICATORS FAILED ON {tick} - {len(self.all_ticks)} remaining ticks')
            self.all_ticks.remove(tick)
            return False

        # Drop NA rows
        df.dropna(inplace=True)


        # Skip if the dataframe is shorter than 200
        if len(df) < 300:
            self.all_ticks.remove(tick)
            return False


        # Sample the dataframe and put in a list
        sample_list = list()
        for i in range(int(len(df)/100)-1):
            sample_list.append(df.iloc[i*100:i*100+200])


        # Process each sample of the dataframe
        valley_dates = list()
        peak_dates = list()
        for sample in sample_list:

            # Process
            peaks, valleys = self.process(sample)
            
            # Check if we found both valleys and peaks - to avoid error
            if len(valleys)>0 and len(peaks)>0:
                valley_dates += list(sample.index[valleys])
                peak_dates += list(sample.index[peaks])
            else:
                continue


        valley_dates = list(set(valley_dates))
        peak_dates = list(set(peak_dates))


        ## Merge the samples
        # Get indicies for valleys/peaks
        valley_indicies = list()
        for valley_date in valley_dates:
            valley_indicies.append(list(df.index).index(valley_date))

        peak_indicies = list()
        for peak_date in peak_dates:
            peak_indicies.append(list(df.index).index(peak_date))
            

        # Determine all sections of buy and sell, peak/valley pair
        peaks, valleys = self.get_sections(peak_indicies, valley_indicies, df)


        # Save or display result
        if self.DEBUG:
            # Plot
            self.plotter(df, peaks, valleys)
            return True

        else:
            ## Save processed data

            # Extract dates for peaks/valleys
            date_buy = df.index[valleys]
            date_sell = df.index[peaks]

            # Save
            self.save(df, tick, date_buy, date_sell)

            self.all_ticks.remove(tick)
            return False


    def process(self, df):
        # Perform lowess on a column, approximation saved as hat
        lowess = sm.nonparametric.lowess(df.RSI_14, range(len(df)), frac=0.01)
        hat = np.array(list(zip(*lowess))[1])

        # Get peaks and valleys
        peaks, valleys = self.get_peaks_valleys(hat)

        # Assert for peaks and valleys
        peaks = self.assert_peaks(peaks, df)
        valleys = self.assert_valleys(valleys, df)

        return peaks, valleys


    def get_peaks_valleys(self, hat):
        ''' FIND PEAKS AND VALLEYS '''

        # Perform scipy signal processing to find peaks/valleys
        _peaks, _ = find_peaks(hat)
        _valleys, _ = find_peaks(-hat, prominence=15)

        # Assert for peaks/valleys
        peaks = list(np.copy(_peaks))
        valleys = list(np.copy(_valleys))

        return peaks, valleys


    def assert_valleys(self, _valleys, df):
        ''' Assert valleys '''
        valleys = list(np.copy(_valleys))
        for valley in _valleys:

            # Remove on RSI
            if df.RSI_14.iloc[valley] > 50:
                valleys.remove(valley)
            elif df.MACDh_12_26_9.iloc[valley] > 0:
                valleys.remove(valley)
            elif df.BBU_signal.iloc[valley] > 0:
                valleys.remove(valley)

        return np.array(valleys)


    def assert_peaks(self, _peaks, df):
        ''' Assert peaks '''
        peaks = list(np.copy(_peaks))
        for peak in _peaks:

            # Remove on MACD
            if any(df[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].iloc[peak] < 0):
                peaks.remove(peak)

            # if df.MACDh_12_26_9.iloc[peak] < 0:
            #     peaks.remove(peak)
            elif df.RSI_14.iloc[peak] < 50:
                peaks.remove(peak)
            elif df.BBU_signal.iloc[peak] < 0:
                peaks.remove(peak)
            
        return np.array(peaks)


    def get_sections(self, peaks, valleys, df):
        ''' Calculates the sections - peak/valley pair '''

        # Sort
        peaks.sort()
        valleys.sort()

        # Convert to array
        peaks = np.array(peaks)
        valleys = np.array(valleys)

        # Go through all valleys to find corresponding peak
        final_peaks = list()
        final_valleys = list()
        for idx, valley in enumerate(valleys):
            if idx == len(valleys)-1:
                break

            # Next valley point
            next_valley = valleys[idx+1]
            
            # Select the peaks between the valley and the next valley, this is called a subsection
            subsection_peaks = peaks[np.logical_and(peaks>valley, peaks<next_valley)]

            # Continue to next valley if no peaks are found in the subsection
            if subsection_peaks.size == 0:
                continue

            # Max RSI for the subsection peaks
            peak_idx_max_rsi = np.argmax(df.RSI_14.iloc[subsection_peaks].to_numpy())
            
            # Append the peak and valley
            final_peaks.append(subsection_peaks[peak_idx_max_rsi])
            final_valleys.append(valley)

        # Convert to array
        final_valleys = np.array(final_valleys)
        final_peaks = np.array(final_peaks)


        ## Assert for valley/peak pairs
        # Calculate the return for each subsection
        subsection_return_thrsh = 0.1
        subsection_return = 1 - df.close.iloc[final_valleys].to_numpy() / df.close.iloc[final_peaks].to_numpy() > subsection_return_thrsh

        # Pick those peaks/valleys that have sufficient return
        final_valleys = final_valleys[subsection_return]
        final_peaks = final_peaks[subsection_return]
        
        return list(final_peaks), list(final_valleys)



    def plotter(self, df, peaks, valleys):
        ''' PLOTS '''

        # Convert to list from array to avoid error
        valleys = list(valleys)
        peaks = list(peaks)

        x = np.array(list(range(len(df.index))))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)

        ''' CLOSE '''
        ax1 = plt.subplot(4,1,1)
        ax1.set_ylabel('CLOSE')
        ax1.yaxis.set_label_position("right")
        ax1.plot(x, df[['close']], linestyle='-', marker='')
        
        # plt.plot(x , hat, linestyle='--')
        plt.plot(x[peaks], df.close[peaks], "v", color='r')
        plt.plot(x[valleys], df.close[valleys], "^", color='g')
        plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')

        ''' MACD '''
        ax2 = plt.subplot(4,1,2, sharex=ax1)
        ax2.set_ylabel('MACD')
        ax2.yaxis.set_label_position("right")
        plt.plot(x, df[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']])

        ''' RSI '''
        ax3 = plt.subplot(4,1,3, sharex=ax1)
        ax3.set_ylabel('RSI')
        ax3.yaxis.set_label_position("right")
        plt.plot(x, df[['RSI_14']], color='black', linewidth=1)
        # plt.plot(x , hat, linestyle='--')
        plt.plot(x[peaks], df.RSI_14[peaks], "v", color='r')
        plt.plot(x[valleys], df.RSI_14[valleys], "^", color='lawngreen')

        ''' BBAND '''
        ax4 = plt.subplot(4,1,4, sharex=ax1)
        ax4.set_ylabel('BBAND')
        ax4.yaxis.set_label_position("right")
        plt.plot(x, df[['BBU_signal']])

        # Adjustments
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        plt.subplots_adjust(bottom=0.18)

        # Show or save
        if "SSH_CONNECTION" in os.environ:
            print('Running from SSH -> fig saved')
            plt.savefig("latest_fig.png")

        else:
            fig.set_figwidth(22)
            fig.set_figheight(10)
            plt.show()


if __name__ == '__main__':
    lp = LP()