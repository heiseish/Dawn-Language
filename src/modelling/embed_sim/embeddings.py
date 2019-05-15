''' Embedding layer for the network'''
from typing import Tuple, List, Optional, Union, Dict, NoReturn

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

__all__ = ['EmbeddingRet', 'EmbeddingSim', 'get_custom_objects']


class EmbeddingRet(keras.layers.Embedding):
    '''Embedding layer with weights returned.'''

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> List[int]:
        return [
            super(EmbeddingRet, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim),
        ]

    def compute_mask(self,
                     inputs: Union[tf.Tensor, List[tf.Tensor]],
                     mask: Optional[Union[tf.Tensor, List[tf.Tensor]]] = None):
        return [
            super(EmbeddingRet, self).compute_mask(inputs, mask),
            None,
        ]

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        return [
            super(EmbeddingRet, self).call(inputs),
            self.embeddings,
        ]


class EmbeddingSim(keras.layers.Layer):
    '''Calculate similarity between features and token embeddings with bias term.'''

    def __init__(self,
                 use_bias: bool = True,
                 initializer: str = 'zeros',
                 regularizer: Optional[str] = None,
                 constraint: Optional[str] = None,
                 **kwargs):
        '''Initialize the layer.
        Args:
            output_dim: Same as embedding output dimension.
            use_bias: Whether to use bias term.
            initializer: Initializer for bias.
            regularizer: Regularizer for bias.
            constraint: Constraint for bias.
            kwargs: Arguments for parent class.
        '''
        super(EmbeddingSim, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.bias = None

    def get_config(self) -> Dict[int, List[int]]:
        ''' Return config of the layers
        Returns
            Dict contaning the config settings
        '''
        config = {
            'use_bias': self.use_bias,
            'initializer': keras.initializers.serialize(self.initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
            'constraint': keras.constraints.serialize(self.constraint),
        }
        base_config = super(EmbeddingSim, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape: Tuple[int, ...]) -> NoReturn:
        ''' Build the layer with the parameters
        Args:
            input_shape (Tuple[int, ...]): input shape of the layer
        '''
        if self.use_bias:
            embed_shape = input_shape[1]
            token_num = embed_shape[0]
            self.bias = self.add_weight(
                shape=(token_num,),
                initializer=self.initializer,
                regularizer=self.regularizer,
                constraint=self.constraint,
                name='bias',
            )
        super(EmbeddingSim, self).build(input_shape)

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        feature_shape, embed_shape = input_shape
        token_num = embed_shape[0]
        return feature_shape[:-1] + (token_num,)

    def compute_mask(self, inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask[0]

    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> tf.Tensor:
        inputs, embeddings = inputs
        outputs = K.dot(inputs, K.transpose(embeddings))
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias)
        return keras.activations.softmax(outputs)


def get_custom_objects() -> Dict[str, keras.layers.Layer]:
    ''' Get custom layer with embedding ret and embedding sim
    Returns:
        an object containing the embedding ret and embedding sim layers
    '''
    return {
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }
