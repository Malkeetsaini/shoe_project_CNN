import torch.nn as nn
from torchvision import models

from configs.config import (
    MAIN_CLASSES,
    SUB_CLASSES,
    TYPE_CLASSES
)

class MultiTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.main_head = nn.Linear(
            in_features,
            len(MAIN_CLASSES)
        )

        self.sub_head = nn.Linear(
            in_features,
            len(SUB_CLASSES)
        )

        self.type_head = nn.Linear(
            in_features,
            len(TYPE_CLASSES)
        )

    def forward(self, x):

        features = self.backbone(x)

        main_out = self.main_head(features)

        sub_out = self.sub_head(features)

        type_out = self.type_head(features)

        return main_out, sub_out, type_out