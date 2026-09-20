import torch 
import torch.nn as nn 

class BiLSTMBlock(nn.Module):
    def __init__(self,input_size,hidden_size):

     super().__init__()


     self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_first=True,
        bidirectional=True      
     )

    def forward(self,x):
        print ("input:",x.shape)

        output, (hidden,cell) = self.lstm(x)

        print("output:",output.shape)   
        print("hidden:",hidden.shape)
        print ("cell:",cell.shape)

        return output