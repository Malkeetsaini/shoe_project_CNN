import torch

from PIL import Image

from models.model import MultiTaskModel

from utils.transforms import val_transform

from configs.config import *

model = MultiTaskModel().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

image_path = "7165046.281.jpg"

image = Image.open(image_path).convert("RGB")

image = val_transform(image)

image = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    main_out, sub_out, type_out = model(image)

main_pred = MAIN_CLASSES[
    main_out.argmax(1).item()
]

sub_pred = SUB_CLASSES[
    sub_out.argmax(1).item()
]

type_pred = TYPE_CLASSES[
    type_out.argmax(1).item()
]

print()

print("Main Category:", main_pred)

print("Subcategory:", sub_pred)

print("Shoe Type:", type_pred)