import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# Model

class baseRNN(keras.Model):
    def __init__(self, activation, L1, L2, feature_dims, lstm_nodes, dense_nodes, window_size, keep_drop, output_dim):
        super(baseRNN, self).__init__()
        self.activation = activation
        self.L1 = regularizers.l1(L1)
        self.L2 = regularizers.l1(L2)
        self.feature_dims = feature_dims
        self.lstm_nodes = lstm_nodes
        self.dense_nodes = dense_nodes
        self.window_size = window_size
        self.keep_drop = keep_drop
        self.output_dim = output_dim
        
        # Model components
        self.dense1 = layers.Dense(self.dense_nodes[0], activation=self.activation, kernel_regularizer=self.L1, input_shape=(None, self.feature_dims))
        self.dropout1 = layers.Dropout(self.keep_drop[0])
        self.lstm1 = layers.Bidirectional(layers.LSTM(self.lstm_nodes[0], return_sequences=True), input_shape=(self.window_size, self.feature_dims))
        self.lstm2 = layers.Bidirectional(layers.LSTM(self.lstm_nodes[0], return_sequences=True), input_shape=(self.window_size, self.feature_dims))
        self.lstm3 = layers.Bidirectional(layers.LSTM(self.lstm_nodes[2]))
        self.dense2 = layers.Dense(self.dense_nodes[1], activation=self.activation, kernel_regularizer=self.L1)
        self.dense3 = layers.Dense(self.dense_nodes[2], activation=self.activation, kernel_regularizer=self.L1)
        self.output_layer = layers.Dense(self.output_dim, activation='softmax')
    
    def __call__(self, x, training=True):
        x = self.dense1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense2(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.dense3(x)
        if training:
            x = self.dropout1(x, training=training)
        
        return self.output_layer(x)