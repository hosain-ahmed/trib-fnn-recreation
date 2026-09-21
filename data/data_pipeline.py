import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# =========================================================
# 1. TEXT CLEANING FUNCTION
# =========================================================

def clean_text(text):
    text = str(text)

    # Remove English alphabets
    text = re.sub(r'[A-Za-z]', '', text)

    # Remove punctuation, emoji, currency signs,
    # pictographic and other unwanted Unicode characters
    text = re.sub(r'[^\u0980-\u09FF0-9\s]', ' ', text)

    # Remove isolated single Bangla characters
    text = re.sub(r'(?<!\S)[\u0980-\u09FF](?!\S)', ' ', text)

    # Collapse repeated spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================================================
# 2. LOAD DATASET
# =========================================================

# Put your actual Excel filename here
df = pd.read_excel("data/raw/bangla_online_comments_dataset.xlsx")

print("Original shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# =========================================================
# 3. KEEP ONLY COMMENT AND LABEL
# =========================================================

df = df[["comment", "label"]].copy()


# =========================================================
# 4. REMOVE MISSING VALUES
# =========================================================

df = df.dropna(subset=["comment", "label"])


# =========================================================
# 5. CLEAN COMMENTS
# =========================================================

df["comment"] = df["comment"].apply(clean_text)


# Remove empty comments after cleaning
df = df[df["comment"].str.strip() != ""]


print("\nAfter cleaning:", df.shape)


# =========================================================
# 6. LABEL MAPPING
# =========================================================

label_map = {
    "not bully": 0,
    "religious": 1,
    "sexual": 2,
    "threat": 3,
    "troll": 4
}

# Keep original text labels for CSV
print("\nLabel distribution:")
print(df["label"].value_counts())


# =========================================================
# 7. TRAIN / TEMP SPLIT
#    70% TRAIN
#    30% TEMP
# =========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# =========================================================
# 8. VALIDATION / TEST SPLIT
#    TEMP → 15% VALIDATION + 15% TEST
# =========================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# =========================================================
# 9. RANDOM OVERSAMPLING
#    ONLY TRAINING DATA
# =========================================================

print("\nTrain distribution before oversampling:")
print(train_df["label"].value_counts())


# Find maximum class size
max_count = train_df["label"].value_counts().max()

balanced_train = []

for label in train_df["label"].unique():

    class_data = train_df[
        train_df["label"] == label
    ]

    class_upsampled = resample(
        class_data,
        replace=True,
        n_samples=max_count,
        random_state=42
    )

    balanced_train.append(class_upsampled)


# Combine all classes
train_df = pd.concat(
    balanced_train
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\nTrain distribution after oversampling:")
print(train_df["label"].value_counts())


# =========================================================
# 10. PRINT DATASET SIZES
# =========================================================

print("\nFinal dataset sizes:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# =========================================================
# 11. CREATE PROCESSED FOLDER
# =========================================================

import os

os.makedirs("data/processed", exist_ok=True)


# =========================================================
# 12. SAVE PROCESSED DATASETS
# =========================================================

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/val.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)


# =========================================================
# 13. FINAL MESSAGE
# =========================================================

print("\nSaved files:")

print("data/processed/train.csv")
print("data/processed/val.csv")
print("data/processed/test.csv")
