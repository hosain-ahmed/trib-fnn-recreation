import torch
import torch.nn as nn

BATCH_SIZE = 4 
SEQ_LEN = 20
VOCAB_SIZE = 1000
EMBED_DIM = 64

x = torch.randint(0,VOCAB_SIZE,(BATCH_SIZE,SEQ_LEN))
print("Input: ", x.shape)

embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBED_DIM)

x = embedding(x)
print("After embedding:", x.shape)

x =x.transpose (1,2)
print("Before Conv1D:", x.shape)

conv = nn.Conv1d(in_channels= 64,out_channels=128,kernel_size=3,padding=1)
x=conv(x)
print("After Conv1D:", x.shape)

bn= nn.BatchNorm1d(128)
relu = nn.ReLU()

x =bn(x)
print("After BatchNorm:", x.shape)

x= relu(x)
print("After ReLU:", x.shape)





