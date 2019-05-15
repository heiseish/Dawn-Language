''' Multihead wrapper layer'''
import copy
from typing import Optional, Dict, Any, List, Tuple, Callable, Union

import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np


class MultiHead(keras.layers.Wrapper):

    def __init__(self,
                 layer: Union[keras.layers.Layer, List[keras.layers.Layer]],
                 layer_num: int = 1,
                 hidden_dim: Optional[Tuple[int, ...]] = None,
                 use_bias: bool = True,
                 reg_index: Optional[int] = None,
                 reg_slice: Optional[int] = None,
                 reg_factor: float = 0.0,
                 **kwargs):
        '''Initialize the wrapper layer.
        Args:
            layer: The layer to be duplicated or a list of layers.
            layer_num: The number of duplicated layers.
            hidden_dim: A linear transformation will be applied to the input data if provided, otherwise the original
                           data will be feed to the sub-layers.
            use_bias: Whether to use bias in the linear transformation.
            reg_index: The index of weights to be regularized.
            reg_slice: The slice indicates which part of the weight to be regularized.
            reg_factor: The weights of the regularization.
            kwargs: Arguments for parent.
        '''
        if type(layer) is list:
            self.layer = layer[0]
            self.layers = layer
            self.layer_num = len(self.layers)
            self.rename = False
        else:
            self.layer = layer
            self.layers = []
            self.layer_num = layer_num
            self.rename = True
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        if reg_index is None or type(reg_index) is list:
            self.reg_index = reg_index
        else:
            self.reg_index = [reg_index]
        if type(reg_slice) is list or reg_index is None:
            self.reg_slice = reg_slice
        else:
            self.reg_slice = [reg_slice] * len(self.reg_index)
        if reg_factor is None or type(reg_factor) is list or reg_index is None:
            self.reg_weight = reg_factor
        else:
            self.reg_weight = [reg_factor] * len(self.reg_index)

        self.W, self.b = None, None
        self.supports_masking = self.layer.supports_masking
        super(MultiHead, self).__init__(self.layer, **kwargs)

    def get_config(self) -> Dict[str, str]:
        slices = None
        if self.reg_slice:
            slices = []
            for interval in self.reg_slice:
                if interval is None:
                    slices.append(None)
                elif type(interval) is slice:
                    slices.append(
                        [interval.start, interval.stop, interval.step])
                else:
                    slices.append([])
                    for sub in interval:
                        slices[-1].append([sub.start, sub.stop, sub.step])
        config = {
            'layers': [],
            'hidden_dim': self.hidden_dim,
            'use_bias': self.use_bias,
            'reg_index': self.reg_index,
            'reg_slice': slices,
            'reg_factor': self.reg_weight,
        }
        for layer in self.layers:
            config['layers'].append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config(),
            })
        base_config = super(MultiHead, self).get_config()
        base_config.pop('layer')
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(
            cls: Callable[[keras.layers.
                           Layer, List[Optional[List[Any]]], Any], Any],
            config: object,
            custom_objects: Optional[object] = None) -> Any:
        reg_slice = config.pop('reg_slice')
        if reg_slice is not None:
            slices = []
            for interval in reg_slice:
                if interval is None:
                    slices.append(None)
                elif type(interval[0]) is list:
                    slices.append([])
                    for sub in interval:
                        slices[-1].append(slice(sub[0], sub[1], sub[2]))
                    slices[-1] = tuple(slices[-1])
                else:
                    slices.append(slice(interval[0], interval[1], interval[2]))
            reg_slice = slices
        layers = [
            keras.layers.deserialize(layer, custom_objects=custom_objects)
            for layer in config.pop('layers')
        ]
        return cls(layers, reg_slice=reg_slice, **config)

    def build(self, input_shape=Tuple[int, ...]) -> Tuple[int, ...]:
        if type(input_shape) == list:
            self.input_spec = list(
                map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        else:
            self.input_spec = keras.engine.InputSpec(shape=input_shape)
        if not self.layers:
            self.layers = [
                copy.deepcopy(self.layer) for _ in range(self.layer_num)
            ]
        if self.hidden_dim is not None:
            self.W = self.add_weight(
                shape=(input_shape[-1], self.hidden_dim * self.layer_num),
                name='{}_W'.format(self.name),
                initializer=keras.initializers.get('uniform'),
            )
            if self.use_bias:
                self.b = self.add_weight(
                    shape=(self.hidden_dim * self.layer_num,),
                    name='{}_b'.format(self.name),
                    initializer=keras.initializers.get('zeros'),
                )
            input_shape = input_shape[:-1] + (self.hidden_dim,)
        for i, layer in enumerate(self.layers):
            if not layer.built:
                if self.rename:
                    layer.name = layer.name + '_%d' % (i + 1)
                layer.build(input_shape)
        if self.reg_index:
            for i, (index, interval, weight) in enumerate(
                    zip(self.reg_index, self.reg_slice, self.reg_weight)):
                weights = []
                if type(interval) is slice:
                    interval = (interval,)
                for layer in self.layers:
                    if interval is None:
                        weights.append(K.flatten(layer.get_weights()[index]))
                    else:
                        weights.append(
                            K.flatten(layer.get_weights()[index][interval]))
                weights = K.stack(weights)
                self.add_loss(weight * K.sum(
                    K.square(
                        K.dot(weights, K.transpose(weights)) -
                        K.eye(len(self.layers)))))
        super(MultiHead, self).build(input_shape)

    def compute_output_shape(self,
                             input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.hidden_dim is not None:
            input_shape = input_shape[:-1] + (self.hidden_dim,)
        child_output_shape = self.layers[0].compute_output_shape(input_shape)
        return child_output_shape + (self.layer_num,)

    def compute_mask(self,
                     inputs: tensorflow.Tensor,
                     mask: Optional[tensorflow.Tensor] = None
                    ) -> Optional[tensorflow.Tensor]:
        return self.layers[0].compute_mask(inputs, mask)

    def call(self,
             inputs: tensorflow.Tensor,
             training: Optional[Any] = None,
             mask: Optional[tensorflow.Tensor] = None) -> tensorflow.Tensor:
        kwargs = {}
        if keras.utils.generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if keras.utils.generic_utils.has_arg(self.layer.call,
                                             'mask') and mask is not None:
            kwargs['mask'] = mask
        if self.hidden_dim is None:
            outputs = [
                K.expand_dims(layer.call(inputs, **kwargs))
                for layer in self.layers
            ]
        else:
            outputs = []
            for i, layer in enumerate(self.layers):
                begin = i * self.hidden_dim
                end = begin + self.hidden_dim
                transformed = K.dot(inputs, self.W[:, begin:end])
                if self.use_bias:
                    transformed += self.b[begin:end]
                outputs.append(K.expand_dims(layer.call(transformed, **kwargs)))
        return K.concatenate(outputs, axis=-1)

    @property
    def trainable_weights(self) -> np.ndarray:
        weights = self._trainable_weights[:]
        for layer in self.layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self) -> np.ndarray:
        weights = self._non_trainable_weights[:]
        for layer in self.layers:
            weights += layer.non_trainable_weights
        return weights

    @property
    def updates(self) -> List[Any]:
        updates = self._updates
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return []

    def get_updates_for(self,
                        inputs: Optional[object] = None) -> Optional[object]:
        inner_inputs = inputs
        if inputs is not None:
            uid = keras.utils.generic_utils.object_list_uid(inputs)
            if uid in self._input_map:
                inner_inputs = self._input_map[uid]

        updates = self._updates
        for layer in self.layers:
            layer_updates = layer.get_updates_for(inner_inputs)
            layer_updates += super(MultiHead, self).get_updates_for(inputs)
            updates += layer_updates
        return updates

    @property
    def losses(self) -> float:
        losses = self._losses
        for layer in self.layers:
            if hasattr(layer, 'losses'):
                losses += layer.losses
        return losses

    def get_losses_for(self, inputs: keras.layers.Layer = None) -> float:
        if inputs is None:
            losses = []
            for layer in self.layers:
                losses = layer.get_losses_for(None)
            return losses + super(MultiHead, self).get_losses_for(None)
        return super(MultiHead, self).get_losses_for(inputs)
