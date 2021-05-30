import tensorflow as tf
import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
class SimpleRNN:
    def __new__(self, inputChar_shape, inputSentence_shape, optimizer, embedding_type, main_output_lstm_units=64, dropout=0.2, recurrent_dropout_rate=0.2, learning_rate=0.01, momentum=0.9, return_optimizer=False):
        # Define two sets of inputs
        inputChar = Input(shape=inputChar_shape, name="char_input")
        inputSentence = Input(shape=inputSentence_shape, name="sentence_input")
        
        maskedInputChar = layers.Masking(mask_value=0)(inputChar)
        maskedInputSentence = layers.Masking(mask_value=0)(inputSentence)

        if ('fasttext' in embedding_type):
            char_bilstm_output = layers.SimpleRNN(100, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout_rate)(maskedInputChar)
        else:
            char_bilstm_output = layers.SimpleRNN(768, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout_rate)(maskedInputChar)

        # Combine the output of the two branches
        combined = layers.concatenate([char_bilstm_output, maskedInputSentence], axis=1)

        # Main output
        z = layers.SimpleRNN(main_output_lstm_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout_rate)(combined)
        z = layers.TimeDistributed(layers.Dense(3, activation='softmax'))(z)

        # Create model
        model = keras.Model(inputs=[inputChar, inputSentence], outputs=z)
        if optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        if (return_optimizer):
            return model, opt
        else:
            return model