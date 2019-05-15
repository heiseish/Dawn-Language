''' Sequence Weighted Attention layer'''
from typing import Optional, Tuple, Dict, NoReturn, List

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K


class SeqWeightedAttention(keras.layers.Layer):
    '''Y = \text{softmax}(XW + b) X

    See: https://arxiv.org/pdf/1708.00524.pdf
    '''

    def __init__(self,
                 use_bias: bool = True,
                 return_attention: bool = False,
                 **kwargs):
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None
        super(SeqWeightedAttention, self).__init__(**kwargs)

    def get_config(self) -> Dict[str, str]:
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(SeqWeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape: Tuple[int, ...]) -> NoReturn:
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(
                shape=(1,),
                name='{}_b'.format(self.name),
                initializer=keras.initializers.get('zeros'))
        super(SeqWeightedAttention, self).build(input_shape)

    def call(self,
             _input: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None) -> tensorflow.Tensor:
        logits = K.dot(_input, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len),
                    (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask: Optional[tensorflow.Tensor] = None
                    ) -> Optional[List[None]]:
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects() -> Dict[str, keras.layers.Layer]:
        return {'SeqWeightedAttention': SeqWeightedAttention}
