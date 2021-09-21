

import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf

from pathlib import Path
import json



class STA2DB:


    def __init__(self):


        # DB credentials
        db_credentials = json.load(open('db_credentials.json'))


        # Establish onnection and cursor
        self.conn = psycopg2.connect(
            host=db_credentials['DB_HOST'],
            dbname=db_credentials['DB_NAME'],
            user=db_credentials['DB_USER'],
            password=db_credentials['DB_PASSW']
            )
        self.cursor = self.conn.cursor()




    def table_creation(self):

        # self.cursor.execute("DROP TABLE test;")

        # Create table
        table_name = 'trigger'
        self.cursor.execute(f'''
        CREATE TABLE {table_name} 
        (
            trigger_id numeric PRIMARY KEY,
            tick varchar,
            buy_price numeric,
            sell_price numeric,
            buy_date date,
            buy_time time,
            sell_date date,
            sell_time time,
            return numeric,
            return_perc numeric,
            trigger_desc_buy varchar,
            trigger_desc_sell varchar,
            model_confidence_buy numeric,
            model_confidence_sell numeric,
            debug BOOLEAN NOT NULL
        );
        ''')



        # table_name = 'daily_tickers'
        # self.cursor.execute(f'''
        # CREATE TABLE {table_name} 
        # (
        #     id serial PRIMARY KEY,
        #     tick varchar,
        #     close_price numeric,
        #     open_price numeric,
        #     volume integer
            
        # );
        # ''')


        # self.cursor.execute("INSERT INTO test (num, data) VALUES (%s, %s)", (100, "abc'def"))

        self.conn.commit()


    def disconnect(self):
        try:
            self.cursor.close()
            self.conn.close()
            print('Disconnected')
        except:
            pass




if __name__ == '__main__':

    # msft = yf.Ticker("MSFT")
    # a = msft.info
    # '''
    # longName
    # country
    # sector
    # '''
    # print(a)
    # quit()
    

    sta2db = STA2DB()
    # sta2db.table_creation()
    sta2db.disconnect()

    # print('=== EOL: db_sandbox.py ===')