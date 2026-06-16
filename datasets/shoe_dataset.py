import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

#import sys
#sys.path.append("/content/drive/MyDrive/shoe_project")

from configs.config import (
    MAIN_CLASSES,
    SUB_CLASSES,
    TYPE_CLASSES
)

class ShoeDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        main_label = MAIN_CLASSES.index(
            row["main_category"]
        )

        sub_label = SUB_CLASSES.index(
            row["subcategory"]
        )

        type_label = TYPE_CLASSES.index(
            row["shoe_type"]
        )

        print({
            "image": image,
            "main_label": main_label,
            "sub_label": sub_label,
            "type_label": type_label
        })

        return {
            "image": image,
            "main_label": main_label,
            "sub_label": sub_label,
            "type_label": type_label
        }