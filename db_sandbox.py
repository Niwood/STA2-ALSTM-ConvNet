

import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf

from pathlib import Path



class STA2DB:

    def __init__(self):


        # DB parameters
        DB_HOST = '172.16.0.6'
        DB_NAME = 'sta2db'
        DB_USER = 'pi'
        DB_PASSW = '0510'


        # Establish onnection and cursor
        self.conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSW)
        self.cursor = self.conn.cursor()




    def table_creation(self):

        # self.cursor.execute("DROP TABLE test;")

        # Create table
        table_name = 'details_companies'
        self.cursor.execute(f'''
        CREATE TABLE {table_name} 
        (
            id varchar PRIMARY KEY,
            company_name varchar,
            sector varchar,
            country varchar
        );
        ''')

        table_name = 'trigger_buy'
        self.cursor.execute(f'''
        CREATE TABLE {table_name} 
        (
            trigger_id serial PRIMARY KEY,
            tick varchar,
            price numeric,
            date date,
            time time,
            return numeric,
            return_perc numeric
        );
        ''')

        table_name = 'trigger_sell'
        self.cursor.execute(f'''
        CREATE TABLE {table_name} 
        (
            trigger_id serial PRIMARY KEY,
            tick varchar,
            price numeric,
            date date,
            time time,
            return numeric,
            return_perc numeric
        );
        ''')

        table_name = 'daily_tickers'
        self.cursor.execute(f'''
        CREATE TABLE {table_name} 
        (
            id serial PRIMARY KEY,
            tick varchar,
            close_price numeric,
            open_price numeric,
            volume integer
            
        );
        ''')


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
    sta2db.table_creation()
    sta2db.disconnect()

    print('=== EOL: db_sandbox.py ===')