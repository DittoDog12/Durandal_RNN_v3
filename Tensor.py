import tensorflow as tf
import numpy as np
import os
#from tensorflow import keras
#from keras.layers import LSTM, Dense, Dropout, CuDNNLSTM, Embedding
#from keras.models import Sequential

class main:
    def create_model(self, units, vocab_size, batch_size, embedding_dim):
        # Check GPU Availablilty, use Cuda if available
        if tf.test.is_gpu_available():
            rnnLayer = tf.keras.layers.CuDNNLSTM
        else:
            import functools
            rnnLayer = functools.partial(tf.keras.layers.LSTM)
        # Create Model
        model = tf.keras.models.Sequential([
            # Add input layer
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                    batch_input_shape=[batch_size, None]),
            # Add LSTM layer selected by above GPU test
            rnnLayer(units, return_sequences=True, 
            recurrent_initializer='orthogonal', stateful=True),
            # Add Dropout layer to randomly shut off LSTM neurons, assists in preventing overfitting, may not be needed            
            #model.add(Dropout(0.1)),
            # Add output layer
            tf.keras.layers.Dense(vocab_size)
        ])           

        # compile model with optimizer and loss function
        model.compile(loss=self.loss, optimizer=tf.train.AdamOptimizer())
        print("Model Summary: ")
        print(model.summary())
        print("")
        return model

    def loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)