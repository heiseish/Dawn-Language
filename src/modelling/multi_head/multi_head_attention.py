'''Multi-head attention layer. '''
from typing import Optional, Tuple, Dict

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from ..self_attention import ScaledDotProductAttention


class MultiHeadAttention(keras.layers.Layer):
    '''Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    '''

    def __init__(self,
                 head_num: int,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_normal',
                 bias_initializer: str = 'zeros',
                 kernel_regularizer: Optional[str] = None,
                 bias_regularizer: Optional[str] = None,
                 kernel_constraint: Optional[str] = None,
                 bias_constraint: Optional[str] = None,
                 history_only: bool = False,
                 **kwargs):
        '''Initialize the layer.
        Args:
            head_num: Number of heads.
            activation: Activations for linear mappings.
            use_bias: Whether to use bias term.
            kernel_initializer: Initializer for linear mappings.
            bias_initializer: Initializer for linear mappings.
            kernel_regularizer: Regularizer for linear mappings.
            bias_regularizer: Regularizer for linear mappings.
            kernel_constraint: Constraints for linear mappings.
            bias_constraint: Constraints for linear mappings.
            history_only: Whether to only use history in attention layer.
        '''
        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self) -> Dict[str, str]:
        config = {
            'head_num':
            self.head_num,
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
            'history_only':
            self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     input_mask: Optional[tensorflow.Tensor] = None
                    ) -> Optional[tensorflow.Tensor]:
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape: Tuple[int, ...]):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = v[-1]
        if feature_dim % self.head_num != 0:
            raise IndexError(
                'Invalid head number %d with the given input dim %d' %
                (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(q[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(k[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(v[-1], feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num: int) -> np.ndarray:
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[
            1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num: int) -> np.ndarray:
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[
            1], input_shape[2]
        x = K.reshape(x,
                      (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(
            x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num: int) -> np.ndarray:
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self,
             inputs: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None
            ) -> Optional[tensorflow.Tensor]:
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y
