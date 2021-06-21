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

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import Tensor
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import  LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation, Conv1D, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.layers import Multiply, Masking, Reshape, Permute, Attention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential




class Net:


    def __init__(self):
        # Paths
        self.model_folder = Path.cwd() / 'models'
        self.staged_folder = Path.cwd() / 'data' / 'staged'

        # Parameters
        self.model_name = str(int(time.time()))
        self.epochs = 100
        self.batch_size = 16
        self.metrics = ('Precision', 'Recall', AUC(curve='PR'))
        self.lr = 1e-4

        # Sequence
        self.load_data()
        self.compile()
        self.train()
        self.eval()
        self.save()


    def load_data(self):

        ''' LOAD DATA '''
        with open(self.staged_folder / 'staged_x.npy', 'rb') as f:
            X = np.load(f, allow_pickle=True)
        with open(self.staged_folder / 'staged_y.npy', 'rb') as f:
            Y = np.load(f, allow_pickle=True)

        # Split test train
        test_size = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_size)
        
        # Edit format to float32
        self.X_train = np.asarray(self.X_train).astype('float32')
        self.X_test = np.asarray(self.X_test).astype('float32')

        print('Staged data loaded:')
        print('X_train shape: ',self.X_train.shape)
        print('X_test shape: ',self.X_test.shape)
        print('y_train shape: ',self.y_train.shape)
        print('y_test shape: ',self.y_test.shape)
        print('-'*10)


    def compile(self):
        ''' Compile network '''
        dropout_rate = 0.3
        num_conv_filters = 8
        # model.add(Attention(name='attention_weight'))

        opt = Adam(learning_rate=self.lr)

        head = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))

        layer = Conv1D(filters=num_conv_filters, kernel_size=3, strides=1 , padding='same', activation='relu')(head)

        num_blocks_list = [2, 5, 5, 5, 2, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                layer = self.residual_conv1d_block(layer, downsample=(j==0 and i!=0), filters=num_conv_filters)
            num_conv_filters *= 2

        layer = GlobalAveragePooling1D()(layer)

        layer = Dense(32, activation='relu')(layer)
        layer = Dropout(dropout_rate)(layer)

        layer = Dense(8, activation='relu')(layer)
        layer = Dropout(dropout_rate)(layer)

        tail = Dense(3, activation='softmax')(layer)

        self.model = Model(inputs=head, outputs=tail)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=list(self.metrics)
            )


    def relu_batchnorm(self, inputs: Tensor) -> Tensor:
        # Helper function for relu activation and batch norm
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn


    def residual_conv1d_block(self, x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        # Helper function for a residual block
        y = Conv1D(kernel_size=kernel_size,
                strides= (1 if not downsample else 2),
                filters=filters,
                padding="same")(x)
        y = self.relu_batchnorm(y)
        y = Conv1D(kernel_size=kernel_size,
                strides=1,
                filters=filters,
                padding="same")(y)

        if downsample:
            x = Conv1D(kernel_size=1,
                    strides=2,
                    filters=filters,
                    padding="same")(x)
        out = Add()([x, y])
        out = self.relu_batchnorm(out)
        return out


    def train(self):
        ''' Train '''
        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True
            )


    def eval(self):
        ''' Evaluate '''
        eval_metrics = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        for val, met in zip(eval_metrics, ('loss', *self.metrics)):
            print(met, val)
        return eval_metrics


    def save(self):
        ''' Save model '''
        self.model.save(self.model_folder / self.model_name)
        print(f'Model saved as: {self.model_name}')


if __name__ == '__main__':
    net = Net()
    print('=== EOL: model.py ===')