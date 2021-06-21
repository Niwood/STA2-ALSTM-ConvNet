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


def on_click(event):
    
    # GET THE CLICKED DATE
    selected_date = mdates.num2date(event.xdata)
    selected_date = selected_date.replace(tzinfo=None)
    if str(selected_date.year) == '1970':
        return
    
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
    df.dropna(inplace=True)


    # Check what year is the next to process
    all_years = set(df.index.year)

    folder = Path.cwd() / 'data' / 'processed'
    all_files = [x.stem for x in folder.glob('*/')]
    all_years_for_tick = set([int(i.split('_')[1]) for i in all_files if tick in i])

    missing_years = list(all_years.difference(all_years_for_tick))

    if len(missing_years) > 0:
        year = str(missing_years[0])
    else:
        print(f'All years have already been saved for {tick}')
        all_ticks.remove(tick)
        continue

    # Select for a particular year
    df = df[year]

    # Reset all lists
    x_sell = []
    y_sell = []
    x_buy = []
    y_buy = []



    ''' PLOTS '''
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1 = plt.subplot(3,1,1)
    ax1.plot(df.index, df[['close']], linestyle='-', marker='')
    plt.title('VALLEY:[SHIFT+LEFT] - PEAK:[SHIFT+RIGHT] - HOLD CTRL TO REMOVE')

    ax2 = plt.subplot(3,1,2, sharex=ax1)
    plt.plot(df.index, df[['MACD_12_26_9', 'MACDs_12_26_9']])

    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plt.plot(df.index, df[['RSI_14']])

    sell_scatter = ax1.scatter([np.nan],[np.nan], color='r', label='sell')

    cid = fig.canvas.mpl_connect('button_press_event', on_click)







        # def prev(self, event):
        #     print('awd')

    callback = Menu()
    # axprev = plt.axes([0.7, 0.0, 0.1, 0.075])
    axdone = plt.axes([0.81, 0.0, 0.1, 0.075])
    button_done = Button(axdone, 'Save')
    button_done.on_clicked(callback.done)
    # bprev = Button(axprev, 'Previous')
    # bprev.on_clicked(callback.prev)





    multi = MultiCursor(fig.canvas, (ax1, ax2, ax3), color='r', lw=1)
    # ax1.set_xlim([datetime.date(2020, 1, 1), datetime.date(2020, 10, 1)])


    plt.subplots_adjust(bottom=0.18)
    plt.show()





