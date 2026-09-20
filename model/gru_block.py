import torch 
import torch.nn as nn

class GRUBlock(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size =input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    def forward(self,x):
        print("Input:",x.shape)
        output,hidden = self.gru(x)
        print("Output:",output.shape)
        print("Hidden:",hidden.shape)
        return output