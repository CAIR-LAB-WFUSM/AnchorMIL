"""Patch-level tumor scorer used for tumor bulk detection (Sec. 3.6, Fig. 3)."""
import torch
import torch.nn as nn
import torchvision


class TumorScorer(nn.Module):
    """ImageNet ResNet50 -> global avg pool -> FC-512 -> FC-1 -> sigmoid."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(r.children())[:-1])  # up to GAP
        self.head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 1))

    def features(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x))).squeeze(-1)


def tumor_score(prob: torch.Tensor) -> torch.Tensor:
    """TS = |sigma(f(x)) - 0.5| / 0.5  (Eq. 1)."""
    return (prob - 0.5).abs() / 0.5


def select_top_patches(prob: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the top-k patches ranked by TS (descending)."""
    k = min(k, prob.numel())
    return torch.topk(tumor_score(prob), k).indices
