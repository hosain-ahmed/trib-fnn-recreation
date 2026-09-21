import sys
import os
import torch

# Allow importing trib_fnn.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trib_fnn import TriBFNN


# Load training tensors
data_path = "data/processed/train_tensors.pt"

train_data = torch.load(
    data_path,
    map_location="cpu"
)

input_ids = train_data["input_ids"]

print("Input IDs shape:", input_ids.shape)


# Get vocabulary size from BanglaBERT tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "csebuetnlp/banglabert"
)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)


# Create model
model = TriBFNN(
    vocab_size=vocab_size,
    num_classes=5
)


print("Model created successfully!")


# Take only 4 samples for testing
sample_input = input_ids[:4]

print("Sample input shape:", sample_input.shape)


# Forward pass
output = model(sample_input)

print("Output shape:", output.shape)

print("Model test completed successfully!") 
import sys
import os
import torch

# Allow importing trib_fnn.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trib_fnn import TriBFNN


# Load training tensors
data_path = "data/processed/train_tensors.pt"

train_data = torch.load(
    data_path,
    map_location="cpu"
)

input_ids = train_data["input_ids"]

print("Input IDs shape:", input_ids.shape)


# Load BanglaBERT tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "csebuetnlp/banglabert"
)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)


# Create model
model = TriBFNN(
    vocab_size=vocab_size,
    num_classes=5
)

print("Model created successfully!")


# Test with 4 samples
sample_input = input_ids[:4]

print("Sample input shape:", sample_input.shape)


# Forward pass
output = model(sample_input)

print("Output shape:", output.shape)

print("Model test completed successfully!")