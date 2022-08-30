
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

class LSTM_MODEL():
    def __init__(
        self,
        num_layers,
        LSTM_units,
        MAX_LEN,
        vocab,
        learning_rate = 3e-4,
        dropout=0.1
    ):
        super(LSTM_MODEL, self).__init__()
        self.num_layers = num_layers
        self.LSTM_units = LSTM_units
        self.MAX_LEN = MAX_LEN
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.len_vocab = len(vocab)
    def build_model(self):
        print("Building Model...")
        model = keras.Sequential()
        model.add(layers.LSTM(self.LSTM_units, input_shape=(self.MAX_LEN, self.len_vocab),dropout=0.1))
        model.add(layers.RepeatVector(self.MAX_LEN))
        # The decoder RNN could be multiple layers stacked or a single layer.
        for _ in range(self.num_layers):
            model.add(layers.LSTM(self.LSTM_units, return_sequences=True,dropout=self.dropout))

        model.add(layers.Dense(self.len_vocab, activation="softmax"))
        return model
    def get_optimizer(self):
        return optimizers.Adam(learning_rate=self.learning_rate)

    def compile_train_model(self,model,data,vocab,epochs,batch_size,print_summary=True):
        x_train,y_train = data['x_train'],data['y_train']
        x_val,y_val = data['x_val'],data['y_val']
        x_test,y_test = data['x_test'],data['y_test']
        optimizer = self.get_optimizer()
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        if(print_summary):
            model.summary()

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                    patience=5, min_lr=0.00001,verbose=1)
        print("Training...")
        for epoch in range(1, epochs+1):
            print("Iteration", epoch)
            model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=[reduce_lr]
            )
            results=[]
            print(f'Test Accuracy: {sum(results)/len(results)}')

    def test(self,model,data,batch_size):
        x_test,y_test = data['x_test'],data['y_test']
        result = model.evaluate(x_test,y_test,batch_size=batch_size)
        print(f'Test Accuracy: {result[1]*100}')
    
    def save_model(self, model, path):
        model.save(path)
    
    def load_model(self,path):
        model = load_model(path)
        return model
    