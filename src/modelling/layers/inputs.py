''' Get input layers'''
import tensorflow
from tensorflow import keras


def get_inputs(seq_len: int) -> tensorflow.Tensor:
    """Get input layers.

    See: https://arxiv.org/pdf/1810.04805.pdf
    Args
        seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment']
    # , 'Masked']
    return [
        keras.layers.Input(
            shape=(None,),
            name='Input-%s' % name,
        ) for name in names
    ]
