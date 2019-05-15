''' Feedforward layer '''
from typing import Tuple, Optional, NoReturn, Dict

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K


class FeedForward(keras.layers.Layer):
    '''Position-wise feed-forward layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    '''

    def __init__(self,
                 units: Tuple[int, ...],
                 activation: str = 'relu',
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_normal',
                 bias_initializer: str = 'zeros',
                 kernel_regularizer: Optional[str] = None,
                 bias_regularizer: Optional[str] = None,
                 kernel_constraint: Optional[str] = None,
                 bias_constraint: Optional[str] = None,
                 **kwargs):
        '''Initialize the layer.
        Args:
            units: Dimension of hidden units.
            activation: Activation for the first linear transformation.
            use_bias: Whether to use the bias term.
            kernel_initializer: Initializer for kernels.
            bias_initializer: Initializer for kernels.
            kernel_regularizer: Regularizer for kernels.
            bias_regularizer: Regularizer for kernels.
            kernel_constraint: Constraint for kernels.
            bias_constraint: Constraint for kernels.
            kwargs:
        '''
        self.supports_masking = True
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self) -> Dict[str, str]:
        config = {
            'units':
            self.units,
            'activation':
            keras.activations.serialize(self.activation),
            'use_bias':
            self.use_bias,
            'kernel_initializer':
            keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer':
            keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
            keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
            keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
            keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint':
            keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     input_mask: Optional[tensorflow.Tensor] = None
                    ) -> tensorflow.Tensor:
        return input_mask

    def build(self, input_shape: Tuple[int, ...]) -> NoReturn:
        feature_dim = input_shape[-1]
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self,
             _input: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None) -> tensorflow.Tensor:
        h = K.dot(_input, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y
