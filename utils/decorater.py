import time
from datetime import datetime
import sys
import pandas as pd

from tabulate import tabulate


def timer(function):
    # Clock the function
    def wrapper(*args, **kwargs):
        d0 = datetime.now()

        function(*args, **kwargs)

        
        print(tabulate([
            ['Total time', str(datetime.now()-d0)],
            ['Start', str(d0)], 
            ['End', str(datetime.now())]]
            ))

        print(f'EOL: {function.__name__} in {sys.modules[function.__module__].__file__}')
 
    return wrapper


def check_columns(function):
    # For utils.feature_engineer.py
    # Check that we have correct columns in a df
    def wrapper(*args, **kwargs):
        df = args[1]
        # df.columns
        print('----')
    return wrapper