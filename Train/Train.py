import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import  DataLoader
from model.Transformer import Transformer
from tqdm import tqdm
import numpy as np
class Training:
    def __init__(self,train_data,val_data,test_data,exp_vocab,batch_size=1024,epochs=50,learning_rate = 3e-4) -> None:
        super(Training, self).__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.exp_vocab = exp_vocab
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_Tx_model(self,embedding_size,num_heads,num_layers,dropout,max_len,forward_expansion):
        
        src_vocab_size = len(self.exp_vocab.vocab)
        trg_vocab_size = len(self.exp_vocab.vocab)
        src_pad_idx = self.exp_vocab.vocab.stoi["<pad>"]
        model = Transformer(
            embedding_size,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            num_heads,
            num_layers, #Encoder
            num_layers, #Decoder
            forward_expansion,
            dropout,
            max_len,
            self.device,
        ).to(self.device)
        return model
    def get_optimizer(self,model):
        return optim.Adam(model.parameters(), lr=self.learning_rate)

    def get_scheduler(self,optimizer,factor,patience,verbose):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, verbose=verbose
        )

    def get_criterion(self):

        pad_idx = self.exp_vocab.vocab.stoi["<pad>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        return criterion

    def get_model_parameters(self,model):
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {parameters}')

    def train(self):
        model = self.get_Tx_model(embedding_size=256,num_heads=8,num_layers=4,dropout=0.1,max_len=31,forward_expansion=4)
        self.get_model_parameters(model)
        step = 0
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer=optimizer,factor=0.1,patience=10,verbose=True)
        criterion = self.get_criterion()
        train_loss=[]
        for epoch in range(self.epochs):
            print(f"[Epoch {epoch} / {self.epochs}]")
            model.train()
            losses = []
            train_loader=DataLoader(self.train_data,self.batch_size,shuffle=True)
            for batch_idx, batch in tqdm(enumerate(train_loader),total=int(self.train_data.shape[0]/self.batch_size)):
                inp_data = torch.transpose(batch[:,0,:],0,1).to(self.device)
                target = torch.transpose(batch[:,1,:],0,1).to(self.device)
                output = model(inp_data, target[:-1, :])
                output = output.reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)
                optimizer.zero_grad()
                loss = criterion(output, target)
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                step += 1

            print(f'Loss: {sum(losses) / len(losses)}')
            self.test(self.val_data,self.exp_vocab,model,self.device,self.batch_size)
            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)
            train_loss.append(mean_loss)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # save_checkpoint and model
        torch.save(checkpoint,f'Seq2Seq_CP_{self.epochs}.pt')
        torch.save(model, f'Seq2Seq_model_{self.epochs}.pt')
        return train_loss
    
    def test(self,data,exp_vocab,model,device,batch_size):
        val_loader=DataLoader(data,batch_size,shuffle=True)
        results=[]
        losses = []
        inputs=[]
        labels=[]
        outputs=[]
        criterion = self.get_criterion()
        model.eval()
        for batch_idx, batch in tqdm(enumerate(val_loader),total=int(data.shape[0]/batch_size)):
            inp_data = torch.transpose(batch[:,0,:],0,1).to(device)
            target = torch.transpose(batch[:,1,:],0,1).to(device)

            with torch.no_grad():
                output = model(inp_data, target[:-1, :])
            out_seq = torch.transpose(output.argmax(2),0,1).detach().cpu().numpy()
            label = batch[:,1,1:].detach().cpu().numpy()
            input_expression = batch[:,0,1:].detach().cpu().numpy()
            for i in range(output.shape[0]):
                flag=0
                out_ix = np.where(out_seq[i,:]==exp_vocab.vocab.stoi['<eos>'])[0]
                if(len(out_ix)>0):
                    out = out_seq[i,:out_ix[0]]
                else:
                    flag=1
                    out = out_seq[i,:]
                if(flag==1):
                    print("FLAGGED")
                translated_sentence = [exp_vocab.vocab.itos[idx] for idx in out]
                source_sentence = [exp_vocab.vocab.itos[idx] for idx in input_expression[i,:np.where(input_expression[i,:]==exp_vocab.vocab.stoi['<eos>'])[0][0]]]
                target_sentence = [exp_vocab.vocab.itos[idx] for idx in label[i,:np.where(label[i,:]==exp_vocab.vocab.stoi['<eos>'])[0][0]]]
                inputs.append(''.join(source_sentence))
                labels.append(''.join(target_sentence))
                outputs.append(''.join(translated_sentence))
                if(flag==1):
                    print(inputs[-1],labels[-1])
                    print(outputs[-1])
                    flag=0
                if(translated_sentence==target_sentence):
                    results.append(1)
                else:
                    results.append(0)
        
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            losses.append(loss.item())
        loss=sum(losses)/len(losses)
        print('-'*50)
        print(f'Valid Loss: {loss}')
        accuracy=sum(results)/len(results)
        print('-'*50)
        print(f'Accuracy: {accuracy}')
        print('-'*50)

        return inputs,labels,outputs
    