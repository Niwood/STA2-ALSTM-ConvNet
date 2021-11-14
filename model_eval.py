from tokenize import Comment
import pandas as pd
import pandas_ta as ta
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor, RadioButtons, Button
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import glob
from pathlib import Path
import time

# from scipy.stats import zscore
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# import tensorflow as tf
# from tensorflow.keras import callbacks
# from tensorflow import Tensor
# from tensorflow.keras.metrics import AUC
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.layers import  LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation, Conv1D, GlobalAveragePooling1D, Bidirectional
# from tensorflow.keras.layers import Multiply, Masking, Reshape, Permute, Attention
# from tensorflow.keras.regularizers import l2

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.python.data.ops.dataset_ops import BatchDataset, TensorDataset
# from tensorflow.python.eager.function import Function
# from tensorflow.python.framework.ops import default_session
# from tensorflow.python.keras.callbacks import TensorBoard
# from tensorflow.python.ops.gen_batch_ops import batch

# from utils.layer_utils import AttentionLSTM
# from utils.decorater import timer

# import keras_tuner
# from keras_tuner import RandomSearch, BayesianOptimization


from tqdm import tqdm

def about_data():

    raw_folder = Path.cwd() / 'data' / 'raw'
    all_ticks = set([x.stem for x in raw_folder.glob('*.pkl')])

    print('TOTAL TICKS: ',len(all_ticks))

    total_length = 0
    for tick in tqdm(all_ticks):
        df = pd.read_pickle(raw_folder / f'{tick}.pkl')

        total_length += len(df)
    print('TOTAL NUMBER OF DAYS: ', total_length)
    print('TOTAL NUMBER OF YEARS: ', total_length/365)

about_data()