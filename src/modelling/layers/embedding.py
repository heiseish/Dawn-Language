''' Embedding layer layer'''
from typing import Optional, Tuple, NoReturn, Dict

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from ..pos_embd import PositionEmbedding
from ..layer_normalization import LayerNormalization


class TokenEmbedding(keras.layers.Embedding):
    '''Embedding layer with weights returned.'''

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return [
            super(TokenEmbedding, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim)
        ]

    def compute_mask(self, inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

    def call(self, inputs: tf.Tensor) -> Optional[tf.Tensor]:
        return [super(TokenEmbedding, self).call(inputs), self.embeddings]


def get_embedding(inputs: keras.layers.Layer,
                  token_num: int,
                  pos_num: int,
                  embed_dim: Tuple[int, ...],
                  dropout_rate: float = 0.1,
                  trainable: bool = True
                 ) -> Tuple[keras.layers.Layer, np.ndarray]:
    '''Get embedding layer.
    See: https://arxiv.org/pdf/1810.04805.pdf
    Args:
        inputs: Input layers.
        token_num: Number of tokens.
        pos_num: Maximum position.
        embed_dim: The dimension of all embedding layers.
        dropout_rate: Dropout rate.
        trainable: Whether the layers are trainable.
    Returns:
        The merged embedding layer and weights of token embedding.
    '''
    embeddings = [
        TokenEmbedding(
            input_dim=token_num,
            output_dim=embed_dim,
            mask_zero=True,
            trainable=trainable,
            name='Embedding-Token',
        )(inputs[0]),
        keras.layers.Embedding(
            input_dim=2,
            output_dim=embed_dim,
            trainable=trainable,
            name='Embedding-Segment',
        )(inputs[1]),
    ]
    embeddings[0], embed_weights = embeddings[0]
    embed_layer = keras.layers.Add(name='Embedding-Token-Segment')(embeddings)
    embed_layer = PositionEmbedding(
        input_dim=pos_num,
        output_dim=embed_dim,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='Embedding-Position',
    )(embed_layer)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='Embedding-Dropout',
        )(embed_layer)
    else:
        dropout_layer = embed_layer
    norm_layer = LayerNormalization(
        trainable=trainable,
        name='Embedding-Norm',
    )(dropout_layer)
    return norm_layer, embed_weights


class EmbeddingSimilarity(keras.layers.Layer):
    '''Calculate similarity between features and token embeddings with bias term.'''

    def __init__(self,
                 initializer: str = 'zeros',
                 regularizer: Optional[str] = None,
                 constraint: Optional[str] = None,
                 **kwargs):
        '''Initialize the layer.
        Args:
            output_dim: Same as embedding output dimension.
            initializer: Initializer for bias.
            regularizer: Regularizer for bias.
            constraint: Constraint for bias.
            kwargs: Arguments for parent class.
        '''
        super(EmbeddingSimilarity, self).__init__(**kwargs)
        self.supports_masking = True
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self) -> Dict[str, str]:
        config = {
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape: Tuple[int, ...]) -> NoReturn:
        self.bias = self.add_weight(
            shape=(input_shape[1][0],),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            name='bias',
        )
        super(EmbeddingSimilarity, self).build(input_shape)

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[0][:2] + (input_shape[1][0],)

    def compute_mask(self, inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask[0]

    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        inputs, embeddings = inputs
        outputs = K.bias_add(K.dot(inputs, K.transpose(embeddings)), self.bias)
        return keras.activations.softmax(outputs)
