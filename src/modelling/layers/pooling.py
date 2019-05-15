''' Masked global max pool 1D '''
from typing import Optional, Tuple

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K


class MaskedGlobalMaxPool1D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPool1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     mask: Optional[tensorflow.Tensor] = None):
        return None

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:-2] + (input_shape[-1],)

    def call(self,
             inputs: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs -= K.expand_dims((1.0 - mask) * 1e6, axis=-1)
        return K.max(inputs, axis=-2)