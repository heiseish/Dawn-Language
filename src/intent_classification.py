''' Train intent classification for dawn'''
import sys
import codecs
from typing import NoReturn

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, concatenate, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

from src.modelling.bert import load_trained_model_from_checkpoint, Tokenizer
from src.modelling.bert.layers import MaskedGlobalMaxPool1D
from src.util import N_CLASS
from src.preprocess import load_data


def train(save_path: str = 'models/model.h5') -> NoReturn:
    ''' Train intent classification for Dawn based on features extracted from BERT
    Args:
        save_path (str): path to save the model
    '''
    CONFIG_PATH = 'models/LargeBert/bert_config.json'
    CHECKPOINT_PATH = 'models/LargeBert/bert_model.ckpt'
    DICT_PATH = 'models/LargeBert/vocab.txt'

    model = load_trained_model_from_checkpoint(
        CONFIG_PATH,
        CHECKPOINT_PATH,
        training=False,
        trainable=False,
        output_layer_num=4,
    )

    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    pool_layer = MaskedGlobalMaxPool1D(name='Pooling')(
        model.get_layer(name='Encoder-Output').output)
    out = Dense(32, activation='relu', name='Pre-Output')(pool_layer)
    output = Dense(units=N_CLASS, activation='softmax',
                   name='Final-Output')(out)
    model = Model(inputs=[
        model.get_layer(name='Input-Token').input,
        model.get_layer(name='Input-Segment').input
    ],
                  outputs=output)
    model.summary(line_length=120)

    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
    checkpoint = ModelCheckpoint(save_path,
                                 verbose=1,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto')
    x_tokens, x_segments, y_in = load_data(dict_path=DICT_PATH)
    model.fit([x_tokens, x_segments],
              y_in,
              epochs=300,
              batch_size=32,
              callbacks=[checkpoint],
              validation_split=0.3,
              shuffle=True)
    # model.save(save_path)


if __name__ == '__main__':
    train()
