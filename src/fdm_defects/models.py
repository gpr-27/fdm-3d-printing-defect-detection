"""ImageNet-pretrained backbones with the same head replacement + freezing policy as the notebooks.

    alexnet      conv features frozen; the whole classifier (3 FC layers, ~54.6M params) trains
    resnet18     everything frozen except the new fc (512 -> classes)
    resnet50     everything frozen except the new fc (2048 -> classes)
    inception_v3 everything frozen except the new fc and the AuxLogits branch (input 299x299)
"""
import torch.nn as nn
from torchvision import models

DEFAULTS = {  # per-model epoch budget and input size used in the notebooks
    "alexnet": {"epochs": 4, "input_size": 224},
    "resnet18": {"epochs": 4, "input_size": 224},
    "resnet50": {"epochs": 4, "input_size": 224},
    "inception_v3": {"epochs": 10, "input_size": 299},
}


def build_model(name, num_classes, pretrained=True):
    if name == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        for p in m.features.parameters():
            p.requires_grad = False
    elif name in ("resnet18", "resnet50"):
        ctor, w = (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1) if name == "resnet18" else \
                  (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1)
        m = ctor(weights=w if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        for n, p in m.named_parameters():
            if "fc" not in n:
                p.requires_grad = False
    elif name == "inception_v3":
        # pretrained weights imply transform_input=True; keep it when building without weights
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
                                aux_logits=True, init_weights=False, transform_input=True)
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        for n, p in m.named_parameters():
            if "fc" not in n and "AuxLogits" not in n:
                p.requires_grad = False
    else:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(DEFAULTS)}")
    return m


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
