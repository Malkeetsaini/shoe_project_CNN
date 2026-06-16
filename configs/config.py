
import torch
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224

MODEL_PATH = "saved_models/best_model.pth"

train_df = pd.read_csv(TRAIN_CSV)

MAIN_CLASSES = sorted(
    train_df["main_category"].unique()
)

SUB_CLASSES = sorted(
    train_df["subcategory"].unique()
)

TYPE_CLASSES = sorted(
    train_df["shoe_type"].unique()
)