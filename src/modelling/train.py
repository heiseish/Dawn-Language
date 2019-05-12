''' Train a deep learning model to classify intention'''
import json
from typing import Tuple, List

from tensorflow.keras.layers import Dense, Input, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

from src.preprocess.extract import load_data, N_LETTERS
from src.util.dawn_intent import N_CLASS

MAX_LENGTH: int = 100
FEATURE_VECTOR_LENGTH: int = 4096
SAVE_PATH: str = 'model/model.h5'

x_vector, y_vector = load_data()
x_np_vector = np.array(x_vector)
y_np_vecotr = np.array(y_vector)
n_a: int = 128

input_layer = Input(shape=(100, N_LETTERS + 1), name='input_layer')
output_layer = Bidirectional(LSTM(n_a, name='bidirectional_lstm'))(input_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(128, activation='relu', name='dense_1')(output_layer)
output_layer = Dropout(0.5)(output_layer)
output_layer = Dense(N_CLASS, activation='softmax',
                     name='dense_2')(output_layer)
model = Model(inputs=input_layer, outputs=output_layer)
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             verbose=1,
                             monitor='val_loss',
                             save_best_only=True,
                             mode='auto')
model.fit(x_np_vector,
          y_np_vecotr,
          epochs=300,
          batch_size=32,
          callbacks=[checkpoint],
          shuffle=True)
model.save(SAVE_PATH)
