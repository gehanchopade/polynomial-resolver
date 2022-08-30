import sys
import model
import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from Preprocessing.Vocab import Vocabulary
from Preprocessing.Preprocess import Preprocess
from model.Model import LSTM_MODEL
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

def train():
    config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 64} ) 
    sess = tf.compat.v1.Session(config=config) 
    keras.backend.set_session(sess)
    MAX_LEN = 29
    num_layers = 4
    units = 256
    epochs = 30
    batch_size = 1024
    preprocess = Preprocess(path='train.txt',MAX_LEN=MAX_LEN,isTest=False)
    data = preprocess.get_split_data(preprocess.x,preprocess.y)
    lstm = LSTM_MODEL(num_layers=num_layers,LSTM_units=units,MAX_LEN=MAX_LEN,vocab=preprocess.characters)
    model = lstm.build_model()
    lstm.compile_train_model(model,data,preprocess.characters,epochs,batch_size)
    lstm.test()

def load_file(file_path: str):
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    MAX_LEN = 29
    preprocess = Preprocess(path=file_path,MAX_LEN=MAX_LEN,isTest=True)
    return preprocess.factors,preprocess.expansions, preprocess.characters

def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors,model,characters, MAX_LEN):
    vocab = Vocabulary(characters, MAX_LEN)
    x = np.zeros((1, MAX_LEN, len(characters)), dtype=np.int32)
    x[0]=vocab.string_numpy(factors)
    pred = model.predict(x)
    expansion = vocab.numpy_string(pred[0]).strip()
    return expansion


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str,model_path: str, MAX_LEN):
    factors, expansions, characters = load_file(filepath)
    print(model_path)
    model = load_model(model_path)    
    scores = []
    print('-'*50)
    print('PREDICTING...')
    print('-'*50)
    for i in tqdm(range(len(factors))):
        factor=factors[i]
        expansion = expansions[i].strip()
        prediction = predict(factor,model,characters, MAX_LEN)
        scores.append(score(prediction,expansion))
    print('-'*50)
    print(f'Accuracy : {np.mean(scores)}')
    print('-'*50)

if __name__ == "__main__":
    if("-t" not in sys.argv):
        train()
    else:
        model_artifact_path = 'factors_expansion.h5'
        main("test.txt",model_artifact_path,MAX_LEN=29)