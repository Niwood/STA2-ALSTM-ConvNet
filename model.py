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

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow import Tensor
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import  LSTM, Input, Dense, Flatten, BatchNormalization, Concatenate, ReLU, Add, Conv2D, MaxPooling2D, Dropout, Activation, Conv1D, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.layers import Multiply, Masking, Reshape, Permute, Attention
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.data.ops.dataset_ops import BatchDataset, TensorDataset
from tensorflow.python.eager.function import Function
from tensorflow.python.framework.ops import default_session
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.ops.gen_batch_ops import batch

from utils.layer_utils import AttentionLSTM
from utils.decorater import timer

import keras_tuner
from keras_tuner import RandomSearch, BayesianOptimization


gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)


class Net:

    @timer
    def __init__(self):

        # Parameters
        MODEL_DESC = 'lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
        # MODEL_DESC = 'test'
        self.MODEL_NAME = str(int(time.time())) + '_' + MODEL_DESC
        self.EPOCHS = 1000
        self.BATCH_SIZE = 25
        self.TEST_SIZE = 0.1
        self.DATA_SET = 'max'

        # Paths
        self.model_folder = Path.cwd() / 'models'
        self.staged_folder = Path.cwd() / 'data' / 'staged'
        self.checkpoint_filepath = Path.cwd() / 'models' / f'checkpoints' / self.MODEL_NAME

        # Metrics
        self.metrics = ('Precision', 'Recall', AUC(curve='PR'))


    # def make_model(self):
        
        # Tensorboard
        self.tensorboard_callback = callbacks.TensorBoard(log_dir="logs/fit/" + self.MODEL_NAME, histogram_freq=1)
        self.earlystopping_callback = callbacks.EarlyStopping(monitor='val_auc', patience=10, mode='max', min_delta=0.001)
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath / '{epoch:02d}-{val_auc:.3f}',
            save_weights_only=False,
            monitor='val_auc',
            mode='max',
            save_best_only=True)

        # Sequence
        self.load_data()


        # tuner = BayesianOptimization(
        #     self.compile,
        #     objective=keras_tuner.Objective("val_auc", direction="max"),
        #     max_trials=20,
        #     executions_per_trial=1,
        #     overwrite=True,
        #     directory="models",
        #     project_name="baysianopt",
        # )
        # print(tuner.search_space_summary())
        # tuner.search(
        #     train_dataset,
        #     epochs=EPOCHS
        #     callbacks=[
        #         callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.001),
        #         tensorboard_callback
        #         ],
        #     validation_data=test_dataset
        #     )
        # tuner.results_summary()

        self.compile()
        self.train()

        self.eval()
        self.save()





    def compile_simple(self, hp=None):
        ''' Compile simple network '''

        opt = Adam(learning_rate=1e-5)
        dropout_rate = 0.4

        # model_type = hp.Choice("model_type", values=['lstm', 'convnet', 'combined'])
        model_type = 'combined'

        # tail_dense_units = hp.Int("tail_dense_units", min_value=100, max_value=500, step=10)
        tail_dense_units = 20
        lstm_units = 16

        # Head
        head = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))

        if model_type == 'lstm':
            # LSTM
            layer = LSTM(lstm_units, kernel_initializer='he_uniform', return_sequences=True)(head)
            layer = LSTM(lstm_units, kernel_initializer='he_uniform')(layer)

        elif model_type == 'convnet':
            # CONVNET
            layer = Permute((2, 1))(head)
            layer = Conv1D(16, 4, padding='valid', kernel_initializer='he_uniform')(layer)
            layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = self.squeeze_excite_block(layer)
            layer = GlobalAveragePooling1D()(layer)

        elif model_type == 'combined':
            # LSTM
            x = LSTM(lstm_units, kernel_initializer='he_uniform', return_sequences=True)(head)
            x = LSTM(lstm_units, kernel_initializer='he_uniform')(x)

            # CONVNET
            y = Permute((2, 1))(head)
            y = Conv1D(16, 4, padding='valid', kernel_initializer='he_uniform')(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = self.squeeze_excite_block(y)
            y = GlobalAveragePooling1D()(y)

            layer = Concatenate()([x, y])


        layer = Dense(tail_dense_units, activation='relu')(layer)
        layer = Dropout(dropout_rate)(layer)

        
        tail = Dense(3, activation='softmax')(layer)

        model = Model(inputs=head, outputs=tail)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=list(self.metrics)
            )

        # print(self.model.summary())
        return model


    def compile(self, hp=None):
        ''' Compile network '''

        '''
        LAST RUN:

        Trial summary
        Hyperparameters:
        conv_filters: 600
        tail_dense_units: 50
        lstm_units: 950
        Score: 0.833
        '''


        # Hyperparameters

        ''' DROPOUT '''
        # dropout_rate = hp.Choice("dropout", values=[0.3, 0.4, 0.5])
        dropout_rate = 0.5

        ''' CONV KERNAL '''
        conv_kernal_size = 2

        ''' CONV FILTERS '''
        # conv_filters = hp.Int("conv_filters", min_value=400, max_value=700, step=10)
        conv_filters = 1024

        ''' TAIL DENSE LAYER '''
        # tail_dense_units = hp.Int("tail_dense_units", min_value=50, max_value=200, step=10)
        # tail_dense_units = hp.Int("tail_dense_units", min_value=10, max_value=60, step=10)
        tail_dense_units = 50
        
        ''' LSTM '''
        # Best: approx 990 units
        # lstm_units = hp.Int("lstm_units", min_value=100, max_value=1200, step=10)
        # num_lstm_layers = hp.Int("num_lstm_layers", min_value=0, max_value=10, step=2, default=5)
        # lstm_units = hp.Choice("lstm_units", values=[256, 512, 1024])
        num_lstm_layers = 2
        lstm_units = 1024
        # l2_reg_param_lstm = hp.Choice("l2_reg_param_lstm", values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        l2_reg_param_lstm = 1e-3 #1e-3 seems to be good
        

        ''' LEARNING RATE '''
        # learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-5, 1e-6])
        LEARNING_RATE = 1e-7


        # model.add(Attention(name='attention_weight'))

        opt = Adam(learning_rate=LEARNING_RATE)

        head = Input(shape=(self.X_train.shape[1],self.X_train.shape[2]))

        x = Masking()(head)
        # x = LSTM(hp.Int(units="units", min_value=32, max_value=512, step=32), return_sequences=True)(x)
        # x = LSTM(lstm_units, return_sequences=True)(x)
        # x = LSTM(
        #     hp.Int("lstm_units", min_value=8, max_value=512, step=32)
        #     )(x) #best 360

        # Best for num_lstm_layers = 0
        for _ in range(num_lstm_layers):
            x = LSTM(
                lstm_units,
                kernel_initializer='he_uniform',
                return_sequences=True,
                kernel_regularizer=l2(l2_reg_param_lstm)
                )(x)


        x = LSTM(lstm_units, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_reg_param_lstm))(x)

        # x = AttentionLSTM(8)(x)
        # x = Attention(name='attention_weight')(x)
        # x = Dropout(dropout_rate)(x)


        y = Permute((2, 1))(head)
        y = Conv1D(conv_filters, conv_kernal_size*4, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(2*conv_filters, conv_kernal_size*2, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(conv_filters, conv_kernal_size, padding='valid', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        z = Concatenate()([x, y])

        layer = Dense(tail_dense_units, activation='relu')(z)
        layer = Dropout(dropout_rate)(layer)

        tail = Dense(3, activation='softmax')(z)

        
        
        model = Model(inputs=head, outputs=tail)
        model.compile(
            loss='categorical_crossentropy',
            # loss='binary_crossentropy',
            optimizer=opt,
            metrics=list(self.metrics)
            )

        # print(self.model.summary())
        self.model = model
        return model


    def relu_batchnorm(self, inputs) -> Tensor:
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


    def load_data(self):

        ''' LOAD DATA '''
        with open(self.staged_folder / f'staged_x_{self.DATA_SET}.npy', 'rb') as f:
            X = np.load(f, allow_pickle=True)
        with open(self.staged_folder / f'staged_y_{self.DATA_SET}.npy', 'rb') as f:
            Y = np.load(f, allow_pickle=True)


        # Split test train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.TEST_SIZE, shuffle=True)
        
        # Edit format to float32
        self.X_train = np.asarray(self.X_train).astype('float32')
        self.X_test = np.asarray(self.X_test).astype('float32')

        print('Staged data loaded:')
        print('X_train shape: ',self.X_train.shape)
        print('X_test shape: ',self.X_test.shape)
        print('y_train shape: ',self.y_train.shape)
        print('y_test shape: ',self.y_test.shape)
        print('-'*10)


        # Convert to tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))

        self.train_dataset = train_dataset.batch(self.BATCH_SIZE)
        self.test_dataset = test_dataset.batch(self.BATCH_SIZE)




    def train(self):
        ''' Train '''

        # class_weight = {0: 1.,
        #                 1: 1.,
        #                 2: 1.}

        self.model.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=self.EPOCHS,
            verbose=True,
            callbacks = [self.tensorboard_callback, self.earlystopping_callback, self.checkpoint_callback]
            # class_weight=class_weight
            )



    def eval(self):
        ''' Evaluate '''
        eval_metrics = self.model.evaluate(self.test_dataset, batch_size=self.BATCH_SIZE)
        for val, met in zip(eval_metrics, ('loss', *self.metrics)):
            print(met, val)
        return eval_metrics


    def save(self):
        ''' Save model '''
        self.model.save(self.model_folder / self.MODEL_NAME)
        print(f'Model saved as: {self.MODEL_NAME}')


if __name__ == '__main__':
    net = Net()