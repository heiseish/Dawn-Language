''' Extract from index '''
from typing import Dict, Tuple, Optional

import tensorflow
from tensorflow import keras


class Extract(keras.layers.Layer):
    '''Extract from index.

    See: https://arxiv.org/pdf/1810.04805.pdf
    '''

    def __init__(self, index: int, **kwargs):
        super(Extract, self).__init__(**kwargs)
        self.index = index
        self.supports_masking = True

    def get_config(self) -> Dict[str, str]:
        config = {
            'index': self.index,
        }
        base_config = super(Extract, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape[:1] + input_shape[2:]

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     mask: Optional[tensorflow.Tensor] = None) -> None:
        return None

    def call(self,
             input: tensorflow.Tensor,
             mask: Optional[tensorflow.Tensor] = None) -> tensorflow.Tensor:
        return input[:, self.index]
