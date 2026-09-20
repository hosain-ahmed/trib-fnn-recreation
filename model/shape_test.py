import torch

from cnn_block import CNNBlock
from bilstm_block import BiLSTMBlock
from gru_block import GRUBlock


print("\n===== CNN TEST =====")

tokens = torch.randint(0, 1000, (4, 20))

cnn = CNNBlock(
    vocab_size=1000,
    embedding_dim=64,
    out_channels=128
)

cnn_output = cnn(tokens)

assert cnn_output.shape == (4, 128, 20)


print("\n===== BiLSTM TEST =====")

features = torch.randn(4, 20, 64)

bilstm = BiLSTMBlock(
    input_size=64,
    hidden_size=128
)

bilstm_output = bilstm(features)

assert bilstm_output.shape == (4, 20, 256)


print("\n===== GRU TEST =====")

features = torch.randn(4, 20, 64)

gru = GRUBlock(
    input_size=64,
    hidden_size=128
)

gru_output = gru(features)

assert gru_output.shape == (4, 20, 128)


print("\nAll shape tests passed!")