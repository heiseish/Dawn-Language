''' Generate output mask based on given mask'''
from typing import Tuple, Optional, Dict

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K


class Masked(keras.layers.Layer):
    '''Generate output mask based on the given mask.

    The inputs for the layer is the original input layer and the masked locations.

    See: https://arxiv.org/pdf/1810.04805.pdf
    '''

    def __init__(self, return_masked: bool = False, **kwargs):
        '''Initialize the layer.

        :param return_masked: Whether to return the merged mask.
        :param kwargs: Arguments for parent class.
        '''
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_masked = return_masked

    def get_config(self) -> Dict[str, str]:
        config = {
            'return_masked': self.return_masked,
        }
        base_config = super(Masked, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.return_masked:
            return [input_shape[0], input_shape[0][:-1]]
        return input_shape[0]

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     mask: Optional[tensorflow.Tensor] = None
                    ) -> tensorflow.Tensor:
        token_mask = K.not_equal(inputs[1], 0)
        return K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)

    def call(self,
             inputs: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None,
             **kwargs) -> tensorflow.Tensor:
        if self.return_masked:
            return [
                inputs[0],
                K.cast(self.compute_mask(inputs, mask), K.floatx())
            ]
        return inputs[0]
