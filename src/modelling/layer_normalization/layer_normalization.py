''' Normalization layer '''
from typing import Optional, Dict, Union, Tuple, Any, NoReturn

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K


class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: Optional[int] = None,
                 gamma_initializer: str = 'ones',
                 beta_initializer: str = 'zeros',
                 gamma_regularizer: Optional[str] = None,
                 beta_regularizer: Optional[str] = None,
                 gamma_constraint: Optional[str] = None,
                 beta_constraint: Optional[str] = None,
                 **kwargs):
        '''Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        Args
            center: Add an offset parameter if it is True.
            scale: Add a scale parameter if it is True.
            epsilon: Epsilon for calculating variance.
            gamma_initializer: Initializer for the gamma weight.
            beta_initializer: Initializer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            kwargs:
        '''
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self) -> Dict[str, Union[str, int]]:
        ''' Merge the base config and config of the current layer
        Returns:
            A dictionary containing the merged configs
        '''
        config = {
            'center':
            self.center,
            'scale':
            self.scale,
            'epsilon':
            self.epsilon,
            'gamma_initializer':
            keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer':
            keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer':
            keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer':
            keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint':
            keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint':
            keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     input_mask: Optional[tensorflow.Tensor] = None
                    ) -> Optional[tensorflow.Tensor]:
        return input_mask

    def build(self, input_shape: Tuple[int, ...]) -> NoReturn:
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs: tensorflow.Tensor,
             training: Optional[Any] = None) -> tensorflow.Tensor:
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
