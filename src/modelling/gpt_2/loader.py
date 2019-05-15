''' load pretrained model'''
from typing import NoReturn, Optional, Callable
import os
import json

import tensorflow as tf
import requests
from tqdm import tqdm
import numpy as np

from .model import get_model

__all__ = ['load_trained_model_from_checkpoint', 'download_gpt2']


def download_gpt2(model_name: str = '345M') -> NoReturn:
    '''Downloads the GPT-2 model into the current directory
    from Google Cloud Storage.
    Adapted from https://github.com/openai/gpt-2/blob/master/download_model.py
    '''

    subdir = os.path.join('models', model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in [
            'checkpoint', 'encoder.json', 'hparams.json',
            'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
            'model.ckpt.meta', 'vocab.bpe'
    ]:

        r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir +
                         "/" + filename,
                         stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100,
                      desc="Fetching " + filename,
                      total=file_size,
                      unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def checkpoint_loader(checkpoint_file: str) -> Callable[[str], np.ndarray]:
    ''' Load a model weights
    Args:
        checkpoint_file (str): path to the checkpint file
    Returns:
        Return a function that takes in the name of the model and 
            return an np array contains the weights of moel
    '''

    def _loader(name: str) -> np.ndarray:
        ''' Load the model weights
        Args:
            name (str): name of the model
        Returns:
            a numpy array representing the weight of the checkpoint
        '''
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def load_trained_model_from_checkpoint(config_path: str,
                                       checkpoint_path: str,
                                       seq_len: Optional[int] = None,
                                       batch_size: Optional[int] = None,
                                       fixed_input_shape: bool = False
                                      ) -> tf.keras.Model:
    '''Load trained official model from checkpoint.
    Args:
        config_path: The path to the JSON configuration file. (hparams.json)
        checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
        seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
        batch_size: Batch size of the model.
        fixed_input_shape: Whether the length of input is fixed. (Needed for TPU training)
    Returns:
        The model.
    '''
    with open(config_path, 'r') as reader:
        config = json.load(reader)
    if seq_len is None:
        n_ctx = config['n_ctx']
    else:
        n_ctx = min(seq_len, config['n_ctx'])
    n_embd = config['n_embd']
    model = get_model(
        n_vocab=config['n_vocab'],
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        batch_size=batch_size,
        fixed_input_shape=fixed_input_shape,
    )

    loader = checkpoint_loader(checkpoint_path)

    model.get_layer(name='Embed-Token').set_weights([
        loader('model/wte:0'),
    ])
    model.get_layer(name='Embed-Token-Pos').set_weights([
        loader('model/wpe:0')[:seq_len, :],
    ])
    for i in range(config['n_layer']):
        model.get_layer(name='Encode-%d-MultiHeadSelfAttention-Norm' %
                        i).set_weights([
                            loader('model/h%d/ln_1/g:0' % i),
                            loader('model/h%d/ln_1/b:0' % i),
                        ])
        kernel = loader('model/h%d/attn/c_attn/w:0' % i)[0]
        bias = loader('model/h%d/attn/c_attn/b:0' % i)
        model.get_layer(name='Encode-%d-MultiHeadSelfAttention' %
                        i).set_weights([
                            kernel[:, :n_embd],
                            bias[:n_embd],
                            kernel[:, n_embd:-n_embd],
                            bias[n_embd:-n_embd],
                            kernel[:, -n_embd:],
                            bias[-n_embd:],
                            loader('model/h%d/attn/c_proj/w:0' % i)[0],
                            loader('model/h%d/attn/c_proj/b:0' % i),
                        ])
        model.get_layer(name='Encode-%d-FeedForward-Norm' % i).set_weights([
            loader('model/h%d/ln_2/g:0' % i),
            loader('model/h%d/ln_2/b:0' % i),
        ])
        model.get_layer(name='Encode-%d-FeedForward' % i).set_weights([
            loader('model/h%d/mlp/c_fc/w:0' % i)[0],
            loader('model/h%d/mlp/c_fc/b:0' % i),
            loader('model/h%d/mlp/c_proj/w:0' % i)[0],
            loader('model/h%d/mlp/c_proj/b:0' % i),
        ])
    model.get_layer(name='Norm').set_weights([
        loader('model/ln_f/g:0'),
        loader('model/ln_f/b:0'),
    ])
    return model
