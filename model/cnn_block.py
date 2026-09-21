import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels):
        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        
        self.conv = nn.Conv1d(
            embedding_dim,
            out_channels,
            kernel_size = 3,
            padding = 1
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU() ### Non linearity 
    
    def forward(self,x):
        print("Input:", x.shape)
        
        x=self.embedding(x)
        print("Embedding:", x.shape)
        
        x = x.transpose(1,2)
        print("Transpose: ")
        
        x = self.conv(x)
        print("Conv1D:", x.shape )
        
        x=self.bn(x)
        print("BatchNorm:", x.shape)
        
        x = self.relu(x)
        print("ReLU:", x.shape)
        
        
        return x
    
    
