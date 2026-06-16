import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets.shoe_dataset import ShoeDataset

from models.model import MultiTaskModel

from utils.transforms import (
    train_transform,
    val_transform
)

from configs.config import *

train_dataset = ShoeDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = ShoeDataset(
    VAL_CSV,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

model = MultiTaskModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_loss = 999999

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)

        main_labels = batch["main_label"].to(DEVICE)

        sub_labels = batch["sub_label"].to(DEVICE)

        type_labels = batch["type_label"].to(DEVICE)

        optimizer.zero_grad()

        main_out, sub_out, type_out = model(images)

        loss1 = criterion(
            main_out,
            main_labels
        )

        loss2 = criterion(
            sub_out,
            sub_labels
        )

        loss3 = criterion(
            type_out,
            type_labels
        )

        loss = loss1 + loss2 + loss3

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss}")

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("Model Saved")