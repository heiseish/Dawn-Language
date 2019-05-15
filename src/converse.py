''' Train conversing for dawn'''
import sys
import codecs
import os
from typing import NoReturn

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, concatenate, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

from src.modelling.gpt_2 import load_trained_model_from_checkpoint, download_gpt2, get_bpe_from_files, generate


def train(model_name: str = '117M',
          save_path: str = 'models/gpt2/model.h5') -> NoReturn:
    ''' Train intent classification for Dawn based on features extracted from BERT
    Args:
        save_path (str): path to save the model
    '''
    if not os.path.exists('models/{}'.format(model_name)):
        download_gpt2(model_name)

    path_to_folder = 'models/{}'.format(model_name)
    config_path = os.path.join(path_to_folder, 'hparams.json')
    checkpoint_path = os.path.join(path_to_folder, 'model.ckpt')
    encoder_path = os.path.join(path_to_folder, 'encoder.json')
    vocab_path = os.path.join(path_to_folder, 'vocab.bpe')

    print('Load model from checkpoint...')
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    model.summary()
    print('Load BPE from files...')
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    print('Generate text...')
    output = generate(model,
                      bpe, ['From the day forth, my arm'],
                      length=20,
                      top_k=1)

    # If you are using the 117M model and top_k equals to 1, then the result will be:
    # "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
    print(output[0])


if __name__ == '__main__':
    train()
