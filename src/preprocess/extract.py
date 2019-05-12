''' Extract data from text files and generate trainable data '''
import unicodedata
import string
import glob
import os
from typing import List, Tuple

import tensorflow as tf
import numpy as np

from src.util.dawn_intent import map_intent_to_number

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


def letter_to_index(letter: str) -> int:
    '''Convert letter to index. eg a -> 1. 0 is reserved for empty letter (for pad_sequence latter)
    Args:
    		letter (str): letter to be converted
    Returns:
    		Integer representing index.
    '''
    return ALL_LETTER.find(letter) + 1


def sentence_to_index(sentence: List[str]) -> List[int]:
    '''Convert sentence to array of letter index
    Args:
    		Sentence (List[str]): Sentence to be converted
    Returns:
    		List[int]: Array of indexes of each character in string
    '''
    return [letter_to_index(c) for c in sentence]


def sentence_to_one_hot_vectors(sentence: List[int]) -> np.ndarray:
    '''Convert sentence of indexes to array of one hot vector
    Args:
    		sentence (List[str]): Sentence to be convert
    Returns:
    		Array of one-hot vectors
    '''
    return tf.one_hot(sentence, N_LETTERS + 1)


def embed(sentence: str) -> np.ndarray:
    ''' Embed the string into 2-D np-array
    Args:
        X (str): string to be convert

    Returns:
        2-D np.ndarray
    '''
    sentence_lower_cased: str = sentence.lower()
    sentence_indexes: List[int] = sentence_to_index(sentence_lower_cased)
    sentence_one_hot_vector: np.ndarray = sentence_to_one_hot_vectors(
        sentence_indexes)
    to_concat = np.zeros((100 - len(sentence_one_hot_vector), N_LETTERS + 1))
    res = np.empty((100, N_LETTERS + 1))
    np.concatenate([sentence_one_hot_vector, to_concat], out=res)
    return res


def load_data() -> Tuple[List[List[int]], np.ndarray]:
    ''' Get data from the text files and convert them to trainable data
    Returns:
        Vectorize data elements
    '''
    x_vector: List[int] = []
    y_vector: List[int] = []
    num_classes: int = 0
    for filename in find_files('data/cases/*.txt'):
        num_classes += 1
        category: str = os.path.splitext(os.path.basename(filename))[0]
        lines: List[str] = read_lines(filename)
        for line in lines:
            x_vector.append(embed(line))
            y_vector.append(map_intent_to_number(category))
    new_y: np.ndarray = tf.keras.utils.to_categorical(y_vector)
    return x_vector, new_y
