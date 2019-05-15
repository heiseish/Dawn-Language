''' Extract data from text files and generate trainable data '''
import unicodedata
import string
import glob
import os
import codecs
import random
from typing import List, Tuple, Any

import tensorflow as tf
import numpy as np

from src.util import map_intent_to_number
from src.modelling.bert import Tokenizer

ALL_LETTER: str = string.ascii_letters + " .,;'"
N_LETTERS: int = len(ALL_LETTER)


def find_files(path: str) -> List[str]:
    ''' Return all files in the given path

    Args:
        path (str): path to the folder

    Returns:
        List[str]: list of filenames as string
    '''
    return glob.glob(path)


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427


def unicode_to_ascii(sentence: str) -> str:
    ''' Convert unicode string to ascii string
    Args:
                        semntence (str): unicode string to convert
        Returns:
                        srring: ascii character strings
        '''
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn' and c in ALL_LETTER)


def read_lines(filename: str) -> List[str]:
    ''' Read lines from txt files.
    Returns:
                List[str]: Array in ASCII
    '''
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def load_data(dict_path: str,
              max_len: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ''' Get data from the text files and convert them to trainable data
    Args:
        dict_path: path to tokenizer dictionary
        max_len: max len of the sentence that should be encoded
    Returns:
        Vectorized data elements
    '''
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)

    x_tokens: List[np.ndarray] = []
    x_segments: List[np.ndarray] = []
    y_vector: List[int] = []
    num_classes: int = 0

    for filename in find_files(path='data/cases/*.txt'):
        num_classes += 1
        category: str = os.path.splitext(os.path.basename(filename))[0]
        lines: List[str] = read_lines(filename=filename)
        for line in lines:
            indices, segments = tokenizer.encode(first=line, max_len=512)
            x_tokens.append(indices)
            x_segments.append(segments)
            y_vector.append(map_intent_to_number(intent=category))

    indices = list(range(len(x_tokens)))
    random.shuffle(indices)
    x_tokens_shuffled = [x_tokens[i] for i in indices]
    x_segments_shuffled = [x_segments[i] for i in indices]
    y_vector_shuffled = [y_vector[i] for i in indices]

    y_final = tf.keras.utils.to_categorical(y_vector_shuffled)
    return np.array(x_tokens_shuffled), np.array(x_segments_shuffled), np.array(
        y_final)
