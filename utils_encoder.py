from utils_pcmci import *
from utilities import *
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import SGD, Adam # For some reason Adam is running slow on M1, M2 so using legacy version

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Dense(d_model)
        self.pos_encoding = get_positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

class TimeSeriesTransformer:
    def __init__(self, input_shape, num_layers=2, d_model=64, num_heads=4, dff=128, rate=0.1, 
                 batch_size=32, epochs=1000, lr=0.01, patience=10, optimizer='adam', loss='mse'):
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.optimizer = Adam(learning_rate=lr) if optimizer == 'adam' else SGD(learning_rate=lr)
        self.loss = loss
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        encoder = TransformerEncoder(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads,
                                     dff=self.dff, maximum_position_encoding=self.input_shape[0], rate=self.rate)
        x = encoder(inputs, training=True)
        x = tf.keras.layers.Flatten()(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping=True, verbosity=1):
        if early_stopping:
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, 
                                     validation_data=validation_data, callbacks=[early_stop], verbose = verbosity)
            return history
        else:
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=validation_data)
            return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

def create_sequences(X, y, sequence_length):
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def model_grid_search(X, y, param_grid, n_splits, sequence_length=12, early_stopping=True):
    num_features = X.shape[1]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params = None
    best_val_loss = float('inf')
    best_history = None

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Evaluating parameters: {param_dict}")
        val_scores = []

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, y_train = X[train_index], y[train_index]
            X_val, y_val = X[val_index], y[val_index]

            X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

            input_shape = (sequence_length, num_features)
            ts_model = TimeSeriesTransformer(input_shape, 
                                             num_layers=param_dict['num_layers'], 
                                             d_model=param_dict['d_model'], 
                                             num_heads=param_dict['num_heads'], 
                                             dff=param_dict['dff'], 
                                             rate=param_dict['dropout'], 
                                             batch_size=param_dict['batch_size'], 
                                             epochs=param_dict['epochs'], 
                                             lr=param_dict['lr'], 
                                             patience=param_dict['patience'], 
                                             optimizer=param_dict['optimizer'], 
                                             loss=param_dict['loss'])

            history = ts_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, early_stopping, verbosity=0)
            val_loss = ts_model.evaluate(X_val_seq, y_val_seq)
            val_scores.append(val_loss)

        print(f'Mean validation loss: {np.mean(val_scores)}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_history = history
        
        print(f"Best parameters so far: {best_params} \\ Best validation loss so far: {best_val_loss}")

    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss}")
    return best_params, best_val_loss, best_history

def train_final_model(X, y, X_test, y_test, best_params, sequence_length=12, n_splits=5, train_on_fulldata=True, plot_loss=True, early_stopping=True):
    num_features = X.shape[1]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if train_on_fulldata:
        X_final_train_seq, y_final_train_seq = create_sequences(X, y, sequence_length)
        X_final_test_seq, y_final_test_seq = create_sequences(X_test, y_test, sequence_length)

        input_shape = (sequence_length, num_features)
        ts_model_final_adjusted = TimeSeriesTransformer(input_shape, 
                                                        num_layers=best_params['num_layers'], 
                                                        d_model=best_params['d_model'], 
                                                        num_heads=best_params['num_heads'], 
                                                        dff=best_params['dff'], 
                                                        rate=best_params['dropout'], 
                                                        batch_size=best_params['batch_size'], 
                                                        epochs=best_params['epochs'], 
                                                        lr=best_params['lr'], 
                                                        patience=best_params['patience'], 
                                                        optimizer=best_params['optimizer'], 
                                                        loss=best_params['loss'])

        history_final_adjusted = ts_model_final_adjusted.train(X_final_train_seq, y_final_train_seq, early_stopping)

        test_loss_adjusted = ts_model_final_adjusted.evaluate(X_final_test_seq, y_final_test_seq)
        print(f'Adjusted Test Loss: {test_loss_adjusted}')

        if plot_loss:
            plt.plot(history_final_adjusted.history['loss'], label='train')
            if 'val_loss' in history_final_adjusted.history:
                plt.plot(history_final_adjusted.history['val_loss'], label='val')
            plt.title('Training Loss per Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        return ts_model_final_adjusted, history_final_adjusted, test_loss_adjusted

    if not train_on_fulldata:
        val_scores = []
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, y_train = X[train_index], y[train_index]
            X_val, y_val = X[val_index], y[val_index]

            X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
            X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

            input_shape = (sequence_length, num_features)
            ts_model = TimeSeriesTransformer(input_shape, 
                                             num_layers=best_params['num_layers'], 
                                             d_model=best_params['d_model'], 
                                             num_heads=best_params['num_heads'], 
                                             dff=best_params['dff'], 
                                             rate=best_params['dropout'], 
                                             batch_size=best_params['batch_size'], 
                                             epochs=best_params['epochs'], 
                                             lr=best_params['lr'], 
                                             patience=best_params['patience'], 
                                             optimizer=best_params['optimizer'], 
                                             loss=best_params['loss'])

            history = ts_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, early_stopping)

            val_loss = ts_model.evaluate(X_val_seq, y_val_seq)
            val_scores.append(val_loss)
            print(f'Validation loss for fold {fold+1}: {val_loss}')

        print(f'Mean validation loss: {np.mean(val_scores)}')

        if plot_loss:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='val')
            plt.title('Training and Validation Loss per Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        return ts_model, history, val_scores
    
def load_and_prepare_data(data_config: str, comp: int, scale='standard', split_date='2016-11-30', MA = False):
    d, fulldata, _ = load_varimax_data(variables, seasons_mask, model_name, n_comps, mask)
    print("Fulldata shape = %s" % str(fulldata.shape))

    start_end_date = d[next(iter(d.keys()))]['results']['start_end']
    end_date = pd.to_datetime(start_end_date[1])
    daterange = pd.date_range(start_end_date[0], end_date + pd.DateOffset(months=1), freq='M')

    fulldata_pd = pd.DataFrame(fulldata, columns=[f'c{i}' for i in range(1, fulldata.shape[1]+1)], index=daterange)
    lagged_features = create_lagged_features(fulldata_pd, 5)

    combined_df = pd.concat([fulldata_pd, lagged_features], axis=1).dropna()

    if MA:
        combined_df[matching_comp_MA] = combined_df[matching_comp_MA].rolling(window=3).mean()
        combined_df = combined_df.dropna()

    # combined_df = combined_df.reset_index(drop=True)
    # Define the train and test data based on a specific desired date for the split
    daterange = combined_df.index

    train_size = daterange.get_loc(split_date) # int(len(fulldata_pd) * 0.90)
    train_df = combined_df.iloc[:train_size]
    test_df = combined_df.iloc[train_size:]

    if data_config == 'combined':
        X = train_df.drop(columns=[f'c{comp+1}']).values
        X_test = test_df.drop(columns=[f'c{comp+1}']).values

    elif data_config == 'selected_lasso':
        X = train_df[selected_variables_lasso[comp]['selected_features_names']].values
        X_test = test_df[selected_variables_lasso[comp]['selected_features_names']].values

    elif data_config == 'selected_pcmci':
        X = train_df[selected_variables[comp]['selected_features_names']].values
        X_test = test_df[selected_variables[comp]['selected_features_names']].values
    
    # adding a new variable selection with PCMCI and autoregressive features up to lag 4
    elif data_config == 'selected_pcmci_+autoreg':
        X = train_df[selected_variables[comp]['selected_features_names'] + [f'c{comp+1}_lag_{i}' for i in range(1, 5)]].values
        X_test = test_df[selected_variables[comp]['selected_features_names'] + [f'c{comp+1}_lag_{i}' for i in range(1, 5)]].values

    elif data_config == 'only_components':        
        train_df = train_df.drop(columns=train_df.filter(like='lag').columns)
        test_df = test_df.drop(columns=test_df.filter(like='lag').columns)

        X = train_df.drop(columns=[f'c{comp+1}']).values
        X_test = test_df.drop(columns=[f'c{comp+1}']).values

    else:
        raise ValueError("Invalid data configuration")

    idx = 7 if MA else 5
    y =  train_df[f'c{comp+1}'].values.reshape(-1, 1)
    y_test = test_df[f'c{comp+1}'].values.reshape(-1, 1)
    y2 = fulldata_pd[f'c{comp+1}'][:train_size].values.reshape(-1, 1)
    y_test2 = fulldata_pd[f'c{comp+1}'][(train_size+idx):].values.reshape(-1, 1)

    print("X shape = %s" % str(X.shape), "y shape = %s" % str(y.shape))
    print("X test shape = %s" % str(X_test.shape), "y test shape = %s" % str(y_test.shape))

    if scale == 'standard':
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)
        y_test = scaler_y.transform(y_test)

        scaler_y = StandardScaler()
        y2 = scaler_y.fit_transform(y2)
        y_test2 = scaler_y.transform(y_test2)

    elif scale == 'minmax':
        scaler_X = MinMaxScaler()
        X = scaler_X.fit_transform(X)
        X_test = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()
        y = scaler_y.fit_transform(y)
        y_test = scaler_y.transform(y_test)

        scaler_y = MinMaxScaler()
        y2 = scaler_y.fit_transform(y2)
        y_test2 = scaler_y.transform(y_test2)

    elif scale == None:
        scaler_X = None
        scaler_y = None
        pass
    else:
        raise ValueError("Invalid scaling method")

    return X, y, X_test, y_test, {'y2': y2, 'y_test2': y_test2, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'time': daterange}