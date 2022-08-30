import random
import numpy as np
from Preprocessing.Vocab import Vocabulary
from tqdm import tqdm
class Preprocess:
    def __init__(self,path,MAX_LEN,isTest):
        super(Preprocess, self).__init__()
        self.lines = open(path, encoding='utf-8').read().strip().split('\n')
        # random.shuffle(self.lines)
        self.MAXLEN = MAX_LEN
        self.isTest = isTest
        self.characters, self.factors, self.expansions = self.prepare_raw_data()
        self.x,self.y = self.vectorize_data()
      
    def prepare_raw_data(self):
        factors = []
        expansions =  []
        characters=set({})
        print('Preprocessing data...')
        for i in tqdm(self.lines):
            factor, expansion = self.preprocess_string(i.split('=')[0]),self.preprocess_string(i.split('=')[1])
            factor = ''.join(factor)
            expansion = ''.join(expansion)
            factor = factor + " "*(self.MAXLEN - len(factor))
            expansion = expansion + " "*(self.MAXLEN - len(expansion))
            factor_chars, expansion_chars = set(factor),set(expansion)
            characters.update(characters.union(factor_chars)-characters)
            characters.update(characters.union(expansion_chars)-characters)
            factors.append(factor)
            expansions.append(expansion)
        return characters,factors,expansions
    def vectorize_data(self):
        vocab = Vocabulary(self.characters,self.MAXLEN)
        print("Vectorizing Data")
        x = np.zeros((len(self.factors), self.MAXLEN, len(self.characters)), dtype=np.bool)
        y = np.zeros((len(self.factors), self.MAXLEN, len(self.characters)), dtype=np.bool)
        for i, sentence in enumerate(self.factors):
            x[i] = vocab.string_numpy(sentence)
        for i, sentence in enumerate(self.expansions):
            y[i] = vocab.string_numpy(sentence)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        return x,y
    
        
    def get_split_data(self,x,y):
        split_at = len(x) - len(x) // 5 #80% data for training 20% for validation and testing
        (x_train, x_val) = x[:split_at], x[split_at:]
        (y_train, y_val) = y[:split_at], y[split_at:]

        split_at = len(x_val) - len(x_val) // 2 #10% for validation and 10% for testing
        (x_test, x_val) = x_val[:split_at], x_val[split_at:]
        (y_test, y_val) = y_val[:split_at], y_val[split_at:]
        data={
            'x_train':x_train,
            'x_test':x_test,
            'x_val':x_val,
            'y_train':y_train,
            'y_test':y_test,
            'y_val':y_val
            
        }
        if(self.isTest):
            return {
                'x_test':x,
                'y_test':y,
            }
        return data

    def preprocess_string(self,s):
        trig_dictionary={
        'cos':'!!!',
        'sin':'###',
        'tan':'@@@'
        }
        s=s.replace('cos',trig_dictionary['cos'])
        s=s.replace('sin',trig_dictionary['sin'])
        s=s.replace('tan',trig_dictionary['tan'])
        for i in range(1,len(s)-1):
    #         print(s[i])
            if(s[i].isalpha() and (not s[i-1].isalpha() and not s[i+1].isalpha())):
                
                char = s[i]
                s=s.replace(char,'x')
                break
        if( s[0].isalpha()):
            char = s[0]
            s=s.replace(char,'x')
        if( s[-1].isalpha()):
            char = s[-1]
            s=s.replace(char,'x')
        s=s.replace(trig_dictionary['cos'],'cos')
        s=s.replace(trig_dictionary['sin'],'sin')
        s=s.replace(trig_dictionary['tan'],'tan')
        
        return list(s)