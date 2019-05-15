''' Load Bert pretrained model'''
import json
from typing import Callable, List, Tuple, Dict, Optional, NoReturn

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .bert import get_model

__all__: List[str] = [
    'build_model_from_config',
    'load_model_weights_from_checkpoint',
    'load_trained_model_from_checkpoint',
]


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


def build_model_from_config(config_file: str,
                            training: bool = False,
                            trainable: Optional[int] = None,
                            output_layer_num=1,
                            seq_len: Optional[int] = None
                           ) -> Tuple[tf.keras.Model, Dict[str, str]]:
    '''Build the model from config file.
    Args:
        config_file (str): The path to the JSON configuration file.
        training (bool): If training, the whole model will be returned.
        trainable (bool): Whether the model is trainable.
        seq_len (int) : If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    Returns: 
        Keras model and the config
    '''
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = min(
            seq_len, config['max_position_embeddings'])
    if trainable is None:
        trainable = training

    model = get_model(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=config['max_position_embeddings'],
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        training=training,
        trainable=trainable,
        output_layer_num=output_layer_num,
    )
    if not training:
        inputs, outputs = model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
        )
    else:
        model.build(input_shape=[(None, None), (None, None), (None, None)])
    return model, config


def load_model_weights_from_checkpoint(model: tf.keras.Model,
                                       config: Dict[str, str],
                                       checkpoint_file: str,
                                       training: bool = False) -> NoReturn:
    '''Load trained official model from checkpoint.
    Args:
        model (tf.keras.Model): Built keras model.
        config (object) : Loaded configuration file.
        checkpoint_file (str): The path to the checkpoint files, should end with '.ckpt'.
        training (bool): If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    '''
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')
        [:config['max_position_embeddings'], :],
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
                loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
                loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
                loader('bert/encoder/layer_%d/attention/output/dense/kernel' %
                       i),
                loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
            ])
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention-Norm' %
            (i + 1)).set_weights([
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' %
                    i),
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' %
                       i),
            ])
        model.get_layer(
            name='Encoder-%d-MultiHeadSelfAttention-Norm' %
            (i + 1)).set_weights([
                loader(
                    'bert/encoder/layer_%d/attention/output/LayerNorm/gamma' %
                    i),
                loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' %
                       i),
            ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(
            name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
                loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
                loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
            ])
    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])


def load_trained_model_from_checkpoint(config_file: str,
                                       checkpoint_file: str,
                                       training: bool = False,
                                       trainable: Optional[int] = None,
                                       output_layer_num=1,
                                       seq_len: Optional[int] = None
                                      ) -> tf.keras.Model:
    '''Load trained official model from checkpoint.
    Args:
        config_file (str): The path to the JSON configuration file.
        checkpoint_file (str): The path to the checkpoint files, should end with '.ckpt'.
        training (bool): If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
        trainable (Optional[int]) : Whether the model is trainable. The default value is the same with `training`.
        seq_len (Optional[int]): If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    Returns
        keras model
    '''
    model, config = build_model_from_config(config_file,
                                            training=training,
                                            trainable=trainable,
                                            output_layer_num=output_layer_num,
                                            seq_len=seq_len)
    load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=training)
    return model
