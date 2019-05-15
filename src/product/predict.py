''' Predict the intent of the sentence using the trained model'''
import sys
from typing import Dict

from tensorflow.keras.models import load_model
import numpy as np

from src.preprocess import embed
from src.util import map_number_to_intent

SAVE_PATH: str = 'model/model.h5'
model = load_model(filepath=SAVE_PATH)
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
tokenizer = Tokenizer(token_dict)


def predict(sentence: str) -> Dict[str, str]:
    ''' Predeict the intent of the sentence using the trained model
    Args:
        sentence (str): Sentence to be classified
    Returns:
        A dict contain the intent as well as the confidence of the prediction
    '''
    indices, segments = tokenizer.encode(first=sentence, max_len=512)
    y_out = model.predict([np.array(indices), np.array(segments)])
    y_temp = np.argmax(y_out, axis=1)
    y_final = [map_number_to_intent(z) for z in y_temp][0]
    res = {}
    res['confidence'] = y_out[0][y_temp][0]
    res['intent'] = y_final
    return res


if __name__ == '__main__':
    sentence = sys.argv[1]
    print(sentence)
    print('Prediction of the sentence [{}] is [{}]'.format(
        sentence, predict(sentence)))
