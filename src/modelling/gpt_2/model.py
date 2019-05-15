''' Build gpt 2 model'''
from typing import Callable, Optional

import tensorflow
from tensorflow import keras

from src.modelling.embed_sim import EmbeddingRet, EmbeddingSim
from src.modelling.pos_embd import PositionEmbedding
from src.modelling.layer_normalization import LayerNormalization
from src.modelling.transformer import gelu, attention_builder, feed_forward_builder, get_encoder_component
from src.modelling.transformer import get_custom_objects as get_transformer_custom_objects

__all__ = ['get_model', 'get_custom_objects']


def _wrap_layer(name: str,
                input_layer: keras.layers.Layer,
                build_func: Callable[[tensorflow.Tensor], tensorflow.Tensor],
                trainable: bool = True) -> keras.layers.Layer:
    '''Wrap layers with normalization and residual.
    Args:
        name: Prefix of names for internal layers.
        input_layer: Input layer.
        build_func: A callable that takes the input tensor and generates the output tensor.
        trainable: Whether the layers are trainable.
    Returns:
        Output layer.
    '''
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(input_layer)
    build_output = build_func(normal_layer)
    return keras.layers.Add(name='%s-Add' % name)([input_layer, build_output])


def _get_encoder_component(name: str,
                           input_layer: keras.layers.Layer,
                           head_num: int,
                           hidden_dim: int,
                           attention_activation: Optional[str] = None,
                           feed_forward_activation: str = 'relu',
                           trainable: bool = True) -> keras.layers.Layer:
    '''Multi-head self-attention and feed-forward layer.
    Args:
        name: Prefix of names for internal layers.
        input_layer: Input layer.
        head_num: Number of heads in multi-head self-attention.
        hidden_dim: Hidden dimension of feed forward layer.
        attention_activation: Activation for multi-head self-attention.
        feed_forward_activation: Activation for feed-forward layer.
        trainable: Whether the layers are trainable.
    Returns
        Output layer.
    '''
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        trainable=trainable,
    )
    return feed_forward_layer


def get_model(n_vocab: int,
              n_ctx: int = 1024,
              n_embd: int = 768,
              n_head: int = 12,
              n_layer: int = 12,
              batch_size: Optional[int] = None,
              fixed_input_shape: bool = False) -> keras.Model:
    '''Get basic GPT-2 model.
    Args
        n_vocab: Number of vocabulary tokens.
        n_ctx: The length of each input.
        n_embd: The dimension of embeddings.
        n_head: Number of heads in transformer.
        n_layer: Number of transformer blocks.
        batch_size: Batch size of the model.
        fixed_input_shape: Whether the length of input is fixed. (Needed for TPU training)
    Returns
        The model.
    '''
    if fixed_input_shape:
        input_layer_shape = (batch_size, n_ctx)
    else:
        input_layer_shape = (batch_size, None)
    input_layer = keras.layers.Input(
        batch_shape=input_layer_shape,
        name='Input',
    )

    embed_token, embeddings = EmbeddingRet(
        input_dim=n_vocab,
        output_dim=n_embd,
        mask_zero=False,
        name='Embed-Token',
    )(input_layer)
    embed_token_pos = PositionEmbedding(
        input_dim=n_ctx,
        output_dim=n_embd,
        mode=PositionEmbedding.MODE_ADD,
        name='Embed-Token-Pos',
    )(embed_token)

    last_layer = embed_token_pos
    for i in range(n_layer):
        last_layer = get_encoder_component(
            name='Encode-%d' % i,
            input_layer=last_layer,
            head_num=n_head,
            hidden_dim=n_embd * 4,
            attention_activation=None,
            feed_forward_activation=gelu,
        )

    norm_layer = LayerNormalization(name='Norm',)(last_layer)

    output_layer = EmbeddingSim(
        use_bias=False,
        name='Output',
    )([norm_layer, embeddings])

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
    )
    return model


def get_custom_objects() -> object:
    custom_objects = get_transformer_custom_objects()
    custom_objects['gelu'] = gelu
    custom_objects['PositionEmbedding'] = PositionEmbedding
    return custom_objects
