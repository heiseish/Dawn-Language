# content of test_sample.py
from src.preprocess import load_data

def func(x):
    return x + 1

def test_answer():
    X, Y = load_data()
    assert func(4) == 5