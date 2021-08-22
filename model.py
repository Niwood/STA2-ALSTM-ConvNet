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

from tensorflow.keras import callbacks
from tensorflow import Tensor
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import  LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation, Conv1D, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.layers import Multiply, Masking, Reshape, Permute, Attention

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential

from utils.layer_utils import AttentionLSTM



class Net:


    def __init__(self):
        # Paths
        self.model_folder = Path.cwd() / 'models'
        self.staged_folder = Path.cwd() / 'data' / 'staged'

        # Parameters
        self.model_name = str(int(time.time())) + '_4lstm_' + '_64conv_'
        self.epochs = 300
        self.batch_size = 32
        self.metrics = ('Precision', 'Recall', AUC(curve='PR'))
        self.lr = 1e-5

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_size, shuffle=True)
        
        # Edit format to float32
        self.X_train = np.asarray(self.X_train).astype('float32')
        self.X_test = np.asarray(self.X_test).astype('float32')

        print('Staged data loaded:')
        print('X_train shape: ',self.X_train.shape)
        print('X_test shape: ',self.X_test.shape)
        print('y_train shape: ',self.y_train.shape)
        print('y_test shape: ',self.y_test.shape)
        print('-'*10)


        for i in self.y_test:
            if sum(i) !=1:
                print(i)
                quit()


    def compile(self):
        ''' Compile network '''
        dropout_rate = 0.5
        num_conv_filters = 64
        lstm_units = 4
        # model.add(Attention(name='attention_weight'))

        opt = Adam(learning_rate=self.lr)

        head = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))

        x = Masking()(head)
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = LSTM(lstm_units, return_sequences=True)(x)
        x = LSTM(lstm_units)(x)
        # x = AttentionLSTM(8)(x)
        # x = Attention(name='attention_weight')(x)
        x = Dropout(dropout_rate)(x)


        y = Permute((2, 1))(head)
        y = Conv1D(num_conv_filters, 8, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(2*num_conv_filters, 4, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(num_conv_filters, 2, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        z = Concatenate()([x, y])

        layer = Dense(8, activation='relu')(z)
        layer = Dropout(dropout_rate)(layer)

        tail = Dense(3, activation='softmax')(layer)


        self.tensorboard_callback = callbacks.TensorBoard(log_dir="logs/fit/" + self.model_name, histogram_freq=1)
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


    def squeeze_excite_block(self, input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        '''
        filters = input.shape[-1] # channel_axis = -1 for TF

        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = Multiply()([input, se])
        return se


    def train(self):
        ''' Train '''
        # class_weight = {0: 1.,
        #                 1: 1.,
        #                 2: 1.}


        self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True,
            callbacks = [self.tensorboard_callback]
            # class_weight=class_weight
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