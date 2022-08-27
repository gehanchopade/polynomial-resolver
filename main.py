import sys
import numpy as np
from typing import Tuple
from Train.Train import Training
from Preprocessing.Preprocess import Preprocess
MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

def train():
    preprocess = Preprocess(path='train.txt',samples=1000000)
    train_data,val_data,test_data = preprocess.prepare_train_test_val_data()
    exp = preprocess.build_vocab()
    print(train_data.shape,val_data.shape,test_data.shape)
    train = Training(train_data,val_data,test_data,exp)
    train.train()

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    return factors


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    if("-t" not in sys.argv):
        train()
    else:
        main("test.txt")