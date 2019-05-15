''' BERT MODEL'''
import math
from typing import List, Optional, Callable, Dict, Union, Tuple

import tensorflow as tf
from tensorflow import keras

import tensorflow.keras.backend as K
import numpy as np

from src.modelling.pos_embd import PositionEmbedding
from src.modelling.layer_normalization import LayerNormalization
from src.modelling.transformer import get_encoders
from src.modelling.transformer import get_custom_objects as get_encoder_custom_objects
from src.modelling.layers import (get_inputs, get_embedding, TokenEmbedding,
                                  EmbeddingSimilarity, Masked, Extract)
from src.modelling.multi_head import MultiHeadAttention
from src.modelling.position_wise_feed_forward import FeedForward

__all__: List[str] = [
    'TOKEN_PAD',
    'TOKEN_UNK',
    'TOKEN_CLS',
    'TOKEN_SEP',
    'TOKEN_MASK',
    'gelu',
    'get_model',
    'get_custom_objects',
    'get_base_dict',
    'gen_batch_inputs',
]

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


def gelu(x: np.ndarray) -> np.ndarray:
    ''' Gelu function
    Args:
        x (np.ndarray): input to the function
    Returns:
        gelu result from x
    '''
    if K.backend() == 'tensorflow':
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (
        1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


class Bert(keras.Model):
    ''' Bert model construction
    '''

    def __init__(
            self,
            token_num: int,
            pos_num: int = 512,
            seq_len: int = 512,
            embed_dim: int = 768,
            transformer_num: int = 12,
            head_num: int = 12,
            feed_forward_dim: int = 3072,
            dropout_rate: int = 0.1,
            attention_activation: Optional[tf.keras.layers.Layer] = None,
            feed_forward_activation: Callable[[np.ndarray], np.ndarray] = gelu,
            custom_layers: Optional[tf.keras.layers.Layer] = None,
            training: bool = True,
            trainable: Optional[int] = None,
            lr: float = 1e-4,
            name: str = 'Bert'):
        super().__init__(name=name)
        self.token_num = token_num
        self.pos_num = pos_num
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.transformer_num = transformer_num
        self.head_num = head_num
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.custom_layers = custom_layers
        self.training = training
        self.trainable = trainable
        self.lr = lr

        # build layers
        # embedding
        self.token_embedding_layer = TokenEmbedding(
            input_dim=token_num,
            output_dim=embed_dim,
            mask_zero=True,
            trainable=trainable,
            name='Embedding-Token',
        )
        self.segment_embedding_layer = keras.layers.Embedding(
            input_dim=2,
            output_dim=embed_dim,
            trainable=trainable,
            name='Embedding-Segment',
        )
        self.position_embedding_layer = PositionEmbedding(
            input_dim=pos_num,
            output_dim=embed_dim,
            mode=PositionEmbedding.MODE_ADD,
            trainable=trainable,
            name='Embedding-Position',
        )
        self.embedding_layer_norm = LayerNormalization(
            trainable=trainable,
            name='Embedding-Norm',
        )

        self.encoder_multihead_layers = []
        self.encoder_ffn_layers = []
        self.encoder_attention_norm = []
        self.encoder_ffn_norm = []
        # attention layers
        for i in range(transformer_num):
            base_name = 'Encoder-%d' % (i + 1)
            attention_name = '%s-MultiHeadSelfAttention' % base_name
            feed_forward_name = '%s-FeedForward' % base_name
            self.encoder_multihead_layers.append(
                MultiHeadAttention(
                    head_num=head_num,
                    activation=attention_activation,
                    history_only=False,
                    trainable=trainable,
                    name=attention_name,
                ))
            self.encoder_ffn_layers.append(
                FeedForward(
                    units=feed_forward_dim,
                    activation=feed_forward_activation,
                    trainable=trainable,
                    name=feed_forward_name,
                ))
            self.encoder_attention_norm.append(
                LayerNormalization(
                    trainable=trainable,
                    name='%s-Norm' % attention_name,
                ))
            self.encoder_ffn_norm.append(
                LayerNormalization(
                    trainable=trainable,
                    name='%s-Norm' % feed_forward_name,
                ))

    def call(self, inputs=np.ndarray) -> np.ndarray:
        ''' Wraps call, applying pre- and post-processing steps.
        Args:
            inputs: input tensor
        Returns:
            tensor
        '''
        embeddings = [
            self.token_embedding_layer(inputs[0]),
            self.segment_embedding_layer(inputs[1])
        ]
        embeddings[0], embed_weights = embeddings[0]
        embed_layer = keras.layers.Add(
            name='Embedding-Token-Segment')(embeddings)
        embed_layer = self.position_embedding_layer(embed_layer)

        if self.dropout_rate > 0.0:
            dropout_layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                name='Embedding-Dropout',
            )(embed_layer)
        else:
            dropout_layer = embed_layer

        embedding_output = self.embedding_layer_norm(dropout_layer)

        def _wrap_layer(name: str,
                        input_layer: tf.keras.layers.Layer,
                        build_func: Callable[[np.ndarray], np.ndarray],
                        norm_layer: tf.keras.layers.Layer,
                        dropout_rate: float = 0.0,
                        trainable: bool = True) -> tf.keras.layers.Layer:
            '''Wrap layers with residual, normalization and dropout.
            Args:
                name (str): Prefix of names for internal layers.
                input_layer (tf.keras.layers.Layer) : Input layer.
                build_func (Callable[[np.ndarray], np.ndarray]): A callable that takes the input tensor and 
                    generates the output tensor.
                dropout_rate (float): Dropout rate.
                trainable (bool): Whether the layers are trainable.
            Returns
                Output layer.
            '''
            build_output = build_func(input_layer)
            if dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=dropout_rate,
                    name='%s-Dropout' % name,
                )(build_output)
            else:
                dropout_layer = build_output
            if isinstance(input_layer, list):
                input_layer = input_layer[0]
            add_layer = keras.layers.Add(name='%s-Add' %
                                         name)([input_layer, dropout_layer])
            normal_layer = norm_layer(add_layer)
            return normal_layer

        last_layer = embedding_output
        output_tensor_list = [last_layer]
        # self attention
        for i in range(self.transformer_num):
            base_name = 'Encoder-%d' % (i + 1)
            attention_name = '%s-MultiHeadSelfAttention' % base_name
            feed_forward_name = '%s-FeedForward' % base_name
            self_attention_output = _wrap_layer(
                name=attention_name,
                input_layer=last_layer,
                build_func=self.encoder_multihead_layers[i],
                norm_layer=self.encoder_attention_norm[i],
                dropout_rate=self.dropout_rate,
                trainable=self.trainable)
            last_layer = _wrap_layer(name=attention_name,
                                     input_layer=self_attention_output,
                                     build_func=self.encoder_ffn_layers[i],
                                     norm_layer=self.encoder_ffn_norm[i],
                                     dropout_rate=self.dropout_rate,
                                     trainable=self.trainable)
            output_tensor_list.append(last_layer)

        return output_tensor_list


def get_model(
        token_num: int,
        pos_num: int = 512,
        seq_len: int = 512,
        embed_dim: int = 768,
        transformer_num: int = 12,
        head_num: int = 12,
        feed_forward_dim: int = 3072,
        dropout_rate: int = 0.1,
        attention_activation: Optional[
            Callable[[np.ndarray], np.ndarray]] = None,
        feed_forward_activation: Callable[[np.ndarray], np.ndarray] = gelu,
        custom_layers: Optional[tf.keras.layers.Layer] = None,
        training: bool = True,
        trainable: Optional[bool] = None,
        output_layer_num=1,
        lr: float = 1e-4
) -> Union[tf.keras.Model, Tuple[keras.layers.Layer, keras.layers.Layer]]:
    '''Build BERT model from hyper parameter

    See: https://arxiv.org/pdf/1810.04805.pdf
    Args:
        token_num (int): Number of tokens.
        pos_num (int): Maximum position.
        seq_len (int): Maximum length of the input sequence or None.
        embed_dim (int): Dimensions of embeddings.
        transformer_num (int): Number of transformers.
        head_num (int): Number of heads in multi-head attention in each transformer.
        feed_forward_dim (int): Dimension of the feed forward layer in each transformer.
        dropout_rate (int): Dropout rate.
        attention_activation Optional[Callable[[np.ndarray], np.ndarray]]: 
            Activation for attention layers.
        feed_forward_activation (Callable[[np.ndarray], np.ndarray]): 
            Activation for feed-forward layers.
        custom_layers (Optional[tf.keras.layers.Layer]): A function that takes the embedding tensor and returns 
            the tensor after feature extraction.
            Arguments such as `transformer_num` and `head_num` will be ignored if `custom_layer` is not `None`.
        training (bool): The built model will be returned if it is `True`, otherwise the input layers 
            and the last feature extraction layer will be returned.
        trainable (Optional[bool]): Whether the model is trainable.
        lr (float): Learning rate.
    Returns
        (Union[tf.keras.Model, Tuple[keras.layers.Layer, keras.layers.Layer]]) The compiled model.
    '''
    if trainable is None:
        trainable = training
    inputs = get_inputs(seq_len=seq_len)
    embed_layer, embed_weights = get_embedding(
        inputs,
        token_num=token_num,
        embed_dim=embed_dim,
        pos_num=pos_num,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    transformed = embed_layer
    if custom_layers is not None:
        kwargs = {}
        if keras.utils.generic_utils.has_arg(custom_layers, 'trainable'):
            kwargs['trainable'] = trainable
        transformed = custom_layers(transformed, **kwargs)
    else:
        transformed = get_encoders(
            encoder_num=transformer_num,
            input_layer=transformed,
            head_num=head_num,
            hidden_dim=feed_forward_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    if not training:
        if output_layer_num > 1:
            if output_layer_num > transformer_num:
                output_layer_num = transformer_num
            model = keras.models.Model(inputs=inputs[:2], outputs=transformed)
            outputs = []
            for i in range(output_layer_num):
                layer = model.get_layer(
                    name='Encoder-{}-FeedForward-Norm'.format(transformer_num -
                                                              i))
                outputs.append(layer.output)
            transformed = keras.layers.Concatenate(name='Encoder-Output')(list(
                reversed(outputs)))
        return inputs[:2], transformed

    mlm_dense_layer = keras.layers.Dense(
        units=embed_dim,
        activation=feed_forward_activation,
        trainable=trainable,
        name='MLM-Dense',
    )(transformed)
    mlm_norm_layer = LayerNormalization(name='MLM-Norm')(mlm_dense_layer)
    mlm_pred_layer = EmbeddingSimilarity(name='MLM-Sim')(
        [mlm_norm_layer, embed_weights])
    masked_layer = Masked(name='MLM')([mlm_pred_layer, inputs[-1]])
    extract_layer = Extract(index=0, name='Extract')(transformed)
    nsp_dense_layer = keras.layers.Dense(
        units=embed_dim,
        activation='tanh',
        trainable=trainable,
        name='NSP-Dense',
    )(extract_layer)
    nsp_pred_layer = keras.layers.Dense(
        units=2,
        activation='softmax',
        trainable=trainable,
        name='NSP',
    )(nsp_dense_layer)
    model = keras.models.Model(inputs=inputs,
                               outputs=[masked_layer, nsp_pred_layer])
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.sparse_categorical_crossentropy,
    )
    return model


def get_custom_objects() -> Dict[str, str]:
    '''Get all custom objects for loading saved models.
    Returns
        An object contains specific attributes
    '''
    custom_objects = get_encoder_custom_objects()
    custom_objects['PositionEmbedding'] = PositionEmbedding
    custom_objects['TokenEmbedding'] = TokenEmbedding
    custom_objects['EmbeddingSimilarity'] = EmbeddingSimilarity
    custom_objects['Masked'] = Masked
    custom_objects['Extract'] = Extract
    custom_objects['gelu'] = gelu
    return custom_objects


def get_base_dict() -> Dict[str, int]:
    '''Get basic dictionary containing special tokens.
    Returns
        A base dict
    '''
    return {
        TOKEN_PAD: 0,
        TOKEN_UNK: 1,
        TOKEN_CLS: 2,
        TOKEN_SEP: 3,
        TOKEN_MASK: 4,
    }


def gen_batch_inputs(sentence_pairs: List[List[str]],
                     token_dict: Dict[str, str],
                     token_list: List[str],
                     seq_len: int = 512,
                     mask_rate: float = 0.15,
                     mask_mask_rate: float = 0.8,
                     mask_random_rate: float = 0.1,
                     swap_sentence_rate: float = 0.5,
                     force_mask: bool = True):
    '''Generate a batch of inputs and outputs for training.
    Args:
        sentence_pairs (List[List[str]]): A list of pairs containing lists of tokens.
        token_dict (Dict[str, str]): The dictionary containing special tokens.
        token_list (List[str]): A list containing all tokens.
        seq_len (int): Length of the sequence.
        mask_rate (float): The rate of choosing a token for prediction.
        mask_mask_rate (float): The rate of replacing the token to `TOKEN_MASK`.
        mask_random_rate (float): The rate of replacing the token to a random word.
        swap_sentence_rate (float): The rate of swapping the second sentences.
        force_mask (bool): At least one position will be masked.
    Returns
        All the inputs and outputs.
    '''
    batch_size = len(sentence_pairs)
    base_dict = get_base_dict()
    unknown_index = token_dict[TOKEN_UNK]
    # Generate sentence swapping mapping
    nsp_outputs = np.zeros((batch_size,))
    mapping = {}
    if swap_sentence_rate > 0.0:
        indices = [
            index for index in range(batch_size)
            if np.random.random() < swap_sentence_rate
        ]
        mapped = indices[:]
        np.random.shuffle(mapped)
        for i in range(len(mapped)):
            if indices[i] != mapped[i]:
                nsp_outputs[indices[i]] = 1.0
        mapping = {indices[i]: mapped[i] for i in range(len(indices))}
    # Generate MLM
    token_inputs, segment_inputs, masked_inputs = [], [], []
    mlm_outputs = []
    for i in range(batch_size):
        first, second = sentence_pairs[i][0], sentence_pairs[mapping.get(i,
                                                                         i)][1]
        segment_inputs.append(([0] * (len(first) + 2) + [1] *
                               (seq_len - (len(first) + 2)))[:seq_len])
        tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
        tokens = tokens[:seq_len]
        tokens += [TOKEN_PAD] * (seq_len - len(tokens))
        token_input, masked_input, mlm_output = [], [], []
        has_mask = False
        for token in tokens:
            mlm_output.append(token_dict.get(token, unknown_index))
            if token not in base_dict and np.random.random() < mask_rate:
                has_mask = True
                masked_input.append(1)
                r = np.random.random()
                if r < mask_mask_rate:
                    token_input.append(token_dict[TOKEN_MASK])
                elif r < mask_mask_rate + mask_random_rate:
                    while True:
                        token = np.random.choice(token_list)
                        if token not in base_dict:
                            token_input.append(token_dict[token])
                            break
                else:
                    token_input.append(token_dict.get(token, unknown_index))
            else:
                masked_input.append(0)
                token_input.append(token_dict.get(token, unknown_index))
        if force_mask and not has_mask:
            masked_input[1] = 1
        token_inputs.append(token_input)
        masked_inputs.append(masked_input)
        mlm_outputs.append(mlm_output)
    inputs = [
        np.asarray(x) for x in [token_inputs, segment_inputs, masked_inputs]
    ]
    outputs = [
        np.asarray(np.expand_dims(x, axis=-1))
        for x in [mlm_outputs, nsp_outputs]
    ]
    return inputs, outputs
