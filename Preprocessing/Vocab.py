import numpy as np

class Vocabulary:
    def __init__(self, chars,MAX_LEN):
        self.MAX_LEN = MAX_LEN
        self.chars = sorted(chars)
        self.stoi,self.itos = self.get_STOI_ITOS(self.chars)
    def get_STOI_ITOS(self,chars):
        return dict((c, i) for i, c in enumerate(chars)),dict((i, c) for i, c in enumerate(chars))
    def string_numpy(self, C):
        x = np.zeros((self.MAX_LEN, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.stoi[c]] = 1
        return x

    def numpy_string(self, x):
        x = x.argmax(axis=-1)
        return "".join(self.itos[x] for x in x)
