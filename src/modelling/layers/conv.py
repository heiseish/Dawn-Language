''' Convolution 1-D layer'''
from typing import Optional

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs: tf.Tensor,
                     mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return mask

    def call(self, inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)
