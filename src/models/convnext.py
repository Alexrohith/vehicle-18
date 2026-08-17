import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights
)


class ConvNeXtClassifier(nn.Module):

    def __init__(
        self,
        num_classes=1,
        pretrained=True
    ):
        super().__init__()

        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None

        self.model = convnext_tiny(
            weights=weights
        )

        # Get classifier input features
        in_features = (
            self.model.classifier[2].in_features
        )

        # Replace original ImageNet classifier
        self.model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(
                in_features,
                eps=1e-6
            ),
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features,
                num_classes
            )
        )

    def forward(self, x):
        return self.model(x)