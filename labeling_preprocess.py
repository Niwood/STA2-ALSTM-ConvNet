import os
import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import numpy as np
import  datetime
import glob
from pathlib import Path
from scipy.signal import find_peaks
import random
import statsmodels.api as sm


def on_click(event):
    
    # GET THE CLICKED DATE
    selected_date = mdates.num2date(event.xdata)
    selected_date = selected_date.replace(tzinfo=None)

    # if str(selected_date.year) == '1970':
    #     return
    
    # FIND CLOSEST IN DF
    _df = df.iloc[df.index.get_loc(selected_date,method='nearest')]
    selected_close = _df.close
    # print('you pressed', selected_date.strftime('%Y-%m-%d'))

    if event.key=='shift' and (event.button is MouseButton.LEFT or MouseButton.RIGHT):
        try:
            if event.button is MouseButton.LEFT:
                x_buy.append(_df.name)
                y_buy.append(selected_close)
                print(f'Buy added at {_df.name}')
                
            elif event.button is MouseButton.RIGHT:
                x_sell.append(_df.name)
                y_sell.append(selected_close)
                print(f'Sell added at {_df.name}')

        except:
            pass
        

    elif event.key == 'control' and (event.button is MouseButton.LEFT or MouseButton.RIGHT):
        
        try:
            idx = y_sell.index(selected_close)
            print(f'Sell removed at {x_sell[idx]}')
            x_sell.pop(idx)
            y_sell.pop(idx)
            
        except:
            pass

        try:
            idx = y_buy.index(selected_close)
            print(f'Buy removed at {x_buy[idx]}')
            x_buy.pop(idx)
            y_buy.pop(idx)
            
        except:
            pass

    print(f'Peaks:{len(x_sell)} | Valleys:{len(x_buy)}')



    ax1.clear()
    ax1.plot(df.index, df[['close']], linestyle='-', marker='')
    ax1.scatter(x_sell,y_sell, color='r')
    ax1.scatter(x_buy,y_buy, color='g')
    plt.sca(ax1)
    plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')
    plt.draw() #redraw



class Menu:
    ind = 0
    def done(self, event):
        print('Done')

        df['target'] = [[1,0,0]]*len(df)
        for x in x_buy:
            df.target.loc[x] = [0,1,0]
        for x in x_sell:
            df.target.loc[x] = [0,0,1]
        print(df)
        processed_foled = Path.cwd() / 'data' / 'processed'
        df.to_csv(processed_foled / f'{tick}_{year}.csv' ,index=True)
        plt.close()


'''GET TEH NEXT TICK'''
raw_folder = Path.cwd() / 'data' / 'raw'
all_ticks = [x.stem for x in raw_folder.glob('*.csv')]
random.shuffle(all_ticks)
# all_ticks = ['TSLA']


while True:
    try:
        tick = all_ticks[0]
    except:
        print('All ticks done.')
        quit()
    print(f'Currently on {tick}')

    df = pd.read_csv(raw_folder / f'{tick}.csv')
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    # TI
    df.ta.macd(fast=12, slow=26, append=True)
    df.ta.rsi(append=True)
    

    # BBAND - Bollinger band upper/lower signal - percentage of how close the hlc is to upper/lower bband
    bband_length = 30
    bband = df.copy().ta.bbands(length=bband_length)
    bband['hlc'] = df.copy().ta.hlc3()

    bbu_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBU_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])
    bbl_signal = (bband['hlc']-bband['BBM_'+str(bband_length)+'_2.0'])/(bband['BBL_'+str(bband_length)+'_2.0'] - bband['BBM_'+str(bband_length)+'_2.0'])

    df['BBU_signal'] = bbu_signal
    df['BBL_signal'] = bbl_signal


    # DROP NA
    df.dropna(inplace=True)


    # Check what year is the next to process
    all_years = set(df.index.year)

    folder = Path.cwd() / 'data' / 'processed'
    all_files = [x.stem for x in folder.glob('*/')]
    all_years_for_tick = set([int(i.split('_')[1]) for i in all_files if tick in i])

    missing_years = list(all_years.difference(all_years_for_tick))
    random.shuffle(missing_years)

    if len(missing_years) > 0:
        year = str(missing_years[0])
        print('YEARS LEFT:', len(missing_years))
    else:
        print(f'All years have already been saved for {tick}')
        all_ticks.remove(tick)
        continue


    # Select for a particular year
    df = df[year]

    # Lowess on RSI
    lowess = sm.nonparametric.lowess(df.RSI_14, range(len(df)), frac=0.03)
    rsi_hat = np.array(list(zip(*lowess))[1])
    # print(type(rsi_hat)), quit()

    # FIND PEAKS AND VALL
    peaks,_ = find_peaks(rsi_hat, prominence=2, distance=10)
    valleys,_ = find_peaks(-rsi_hat, prominence=2, distance=10)

    '''
    ALG

    PEAK:
    - pos BBU_signal (above 1)
    - pos MACDh_12_26_9 for x steps
    - rsi above 60


    VAllEY:
    - neg BBU_signal (below -1)
    - neg MACDh_12_26_9 for x steps
    - rsi below 40

    '''
    macd_trailing_thr = 5
    for peak in peaks:
        if df.BBU_signal[peak] < 1:
            peaks = np.delete(peaks, np.where(peaks == peak))

        # elif df.RSI_14[peak] < 60:
        #     peaks = np.delete(peaks, np.where(peaks == peak))

        # elif not (df.MACDh_12_26_9[peak-macd_trailing_thr:peak] > 0).all():
        #     peaks = np.delete(peaks, np.where(peaks == peak))

    for valley in valleys:
        if df.BBU_signal[valley] > -0.5:
            valleys = np.delete(valleys, np.where(valleys == valley))

        # elif df.RSI_14[valley] > 50:
        #     valleys = np.delete(valleys, np.where(valleys == valley))

        # elif not (df.MACDh_12_26_9[peak-macd_trailing_thr:peak] < 0).all():
        #     valleys = np.delete(valleys, np.where(valleys == valley))


    # Reset all lists
    x_sell = []
    y_sell = []
    x_buy = []
    y_buy = []

    ''' PLOTS '''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)

    ax1 = plt.subplot(4,1,1)
    ax1.plot(df.index, df[['close']], linestyle='-', marker='')
    plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    plt.plot(df.index, df[['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']])

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    plt.plot(df.index, df[['RSI_14']])
    plt.plot(df.index , rsi_hat)
    # print( df.RSI_14.to_list() ), quit()
    # print(peaks)
    # print(df.RSI_14[[1,2,3]])
    plt.plot(df.index[peaks], df.RSI_14[peaks], "x")
    plt.plot(df.index[valleys], df.RSI_14[valleys], "x")

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    plt.plot(df.index, df[['BBU_signal']])

    sell_scatter = ax1.scatter([np.nan],[np.nan], color='r', label='sell')

    cid = fig.canvas.mpl_connect('button_press_event', on_click)



    callback = Menu()
    # axprev = plt.axes([0.7, 0.0, 0.1, 0.075])
    axdone = plt.axes([0.81, 0.0, 0.1, 0.075])
    button_done = Button(axdone, 'Save')
    button_done.on_clicked(callback.done)
    # bprev = Button(axprev, 'Previous')
    # bprev.on_clicked(callback.prev)



    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    multi = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4), color='r', lw=1)
    # ax1.set_xlim([datetime.date(2020, 1, 1), datetime.date(2020, 10, 1)])


    plt.subplots_adjust(bottom=0.18)
    

    if "SSH_CONNECTION" in os.environ:
        print('Running from SSH -> fig saved')
        plt.savefig("latest_fig.png")
        quit()
    else:
        plt.show()





