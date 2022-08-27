import random
from torchtext.legacy.data import Field
import torch
import numpy as np
from tqdm import tqdm
class PreprocessData:
    def __init__(self,path,samples):
        super(PreprocessData, self).__init__()
        self.lines = open(path, encoding='utf-8').read().strip().split('\n')
        random.shuffle(self.lines)
        self.samples=samples
        lines=random.sample(lines,samples)
        self.raw_data=self.lines
        self.vocab_data = [[self.preprocess_string(l.split('=')[0]), self.preprocess_string(l.split('=')[1])] for l in self.lines]
        self.train = [{'src':self.preprocess_string(l.split('=')[0]), 'trg':self.preprocess_string(l.split('=')[1])} for l in self.lines]

    def prepare_train_test_val_data(self):
        max_len=31
        data_list=[]
        exp=self.build_vocab()
        pad_idx=exp.vocab.stoi['<pad>']
        for element in tqdm(self.train):
            src=[exp.vocab.stoi[i] for i in (['<sos>']+element['src']+['<eos>'])]
            trg=[exp.vocab.stoi[i] for i in (['<sos>']+element['trg']+['<eos>'])]
            data_list.append([src+[pad_idx]*(max_len-len(src)),
                            trg+[pad_idx]*(max_len-len(trg))])
        x=torch.tensor(data_list)
        train_data=x[:int(self.samples*0.8),:,:]
        val_data=x[int(self.samples*0.8):int(self.samples*0.9),:,:]
        test_data=x[int(self.samples*0.9):,:,:]

        return train_data,val_data,test_data
    def build_vocab(self):
        def tokenize(text):
            return list(text)
        exp = Field(
            tokenize=tokenize, lower=True, init_token="<sos>", eos_token="<eos>"
        )

        VB = np.array(self.vocab_data)
        VB_src=VB[:,0]
        exp.build_vocab(VB_src, max_size=100000, min_freq=1)
        return exp
    
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