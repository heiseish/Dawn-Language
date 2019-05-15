''' Gelu function'''
import math
import tensorflow.keras.backend as K
import numpy


def gelu(_input: numpy.ndarray) -> numpy.ndarray:
    '''An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    Args:
        Input numpy array
    Returns:
        gelu function result
    '''
    return 0.5 * _input * (1.0 + K.tanh(
        math.sqrt(2.0 / math.pi) * (_input + 0.044715 * K.pow(_input, 3))))
