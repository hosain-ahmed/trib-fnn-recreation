import pandas as pd
import torch
from transformers import AutoTokenizer


# ==========================================
# 1. Load BanglaBERT Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "csebuetnlp/banglabert"
)


# ==========================================
# 2. Load Processed Datasets
# ==========================================

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")


print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ==========================================
# 3. Label Mapping
# ==========================================

label_map = {
    "not bully": 0,
    "religious": 1,
    "sexual": 2,
    "threat": 3,
    "troll": 4
}


# ==========================================
# 4. Tokenization Function
# ==========================================

def tokenize_data(df):

    # Get comments
    texts = df["comment"].astype(str).tolist()

    # Convert text labels to numbers
    labels = df["label"].map(label_map).tolist()

    # Check if any label could not be mapped
    if any(label is None for label in labels):
        raise ValueError("Some labels could not be mapped.")

    # BanglaBERT tokenization
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Return tensors
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long)
    }


# ==========================================
# 5. Tokenize Train, Validation and Test
# ==========================================

print("\nStarting tokenization...")

train_data = tokenize_data(train_df)
val_data = tokenize_data(val_df)
test_data = tokenize_data(test_df)


# ==========================================
# 6. Save Tensor Files
# ==========================================

torch.save(
    train_data,
    "data/processed/train_tensors.pt"
)

torch.save(
    val_data,
    "data/processed/val_tensors.pt"
)

torch.save(
    test_data,
    "data/processed/test_tensors.pt"
)


# ==========================================
# 7. Display Results
# ==========================================

print("\nTokenization completed!")

print(
    "Train Input IDs shape:",
    train_data["input_ids"].shape
)

print(
    "Train Attention Mask shape:",
    train_data["attention_mask"].shape
)

print(
    "Train Labels shape:",
    train_data["labels"].shape
)

print(
    "Validation Input IDs shape:",
    val_data["input_ids"].shape
)

print(
    "Test Input IDs shape:",
    test_data["input_ids"].shape
)


# ==========================================
# 8. Confirm Saved Files
# ==========================================

print("\nSaved files:")

print("data/processed/train_tensors.pt")
print("data/processed/val_tensors.pt")
print("data/processed/test_tensors.pt")
