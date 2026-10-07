import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes: int, pretrained: bool = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Engineering symbols are grayscale.
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
