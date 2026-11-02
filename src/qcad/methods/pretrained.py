from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data import to_tensor
from .autoencoder import gaussian_blur
from .base import AnomalyMethod, Prediction

BACKBONES = {"resnet18": "resnet18.tv_in1k", "wide_resnet50_2": "wide_resnet50_2.tv_in1k"}
LAYERS = ("layer1", "layer2", "layer3", "layer4")
PADIM_FEATURES = {"resnet18": 100, "wide_resnet50_2": 550}
AGGREGATIONS = ("concat", "original")

_BACKBONES: dict = {}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timm_name(backbone: str) -> str:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone: {backbone}")
    return BACKBONES[backbone]


def load_backbone(backbone: str, layers, pretrained: bool = True):
    import timm

    layers = tuple(layers)
    unknown = [layer for layer in layers if layer not in LAYERS]
    if unknown or not layers:
        raise ValueError(f"unknown layers: {unknown}")
    key = (backbone, layers, pretrained)
    if key not in _BACKBONES:
        model = timm.create_model(timm_name(backbone), pretrained=pretrained, features_only=True,
                                  out_indices=[LAYERS.index(layer) + 1 for layer in layers])
        _BACKBONES[key] = model.eval().requires_grad_(False).to(device())
    return _BACKBONES[key]


def backbone_params(backbone: str, layers, pretrained: bool = True) -> int:
    return int(sum(p.numel() for p in load_backbone(backbone, layers, pretrained).parameters()))


def batches(images: np.ndarray, batch_size: int):
    for start in range(0, len(images), batch_size):
        yield to_tensor(images[start:start + batch_size], imagenet_norm=True).to(device())


@torch.no_grad()
def cnn_features(images: np.ndarray, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"),
                 batch_size: int = 32, pretrained: bool = True) -> np.ndarray:
    model = load_backbone(backbone, layers, pretrained)
    out = [torch.cat([f.mean(dim=(2, 3)) for f in model(x)], dim=1).cpu() for x in batches(images, batch_size)]
    return torch.cat(out).numpy().astype(np.float32)


def fit_gaussians(x: torch.Tensor, epsilon: float, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    p, n, d = x.shape
    if n < 2:
        raise ValueError(f"need at least 2 images, got {n}")
    mean = x.mean(dim=1)
    inv = torch.empty(p, d, d, dtype=torch.float32)
    eye = torch.eye(d, dtype=torch.float64, device=device())
    for s in range(0, p, chunk):
        xc = x[s:s + chunk].to(device(), torch.float64)
        xc = xc - xc.mean(dim=1, keepdim=True)
        cov = xc.transpose(1, 2) @ xc / (n - 1) + epsilon * eye
        inv[s:s + chunk] = torch.linalg.inv(cov).float().cpu()
    return mean, inv


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, inv: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    p, n, _ = x.shape
    out = torch.empty(p, n, dtype=torch.float32)
    for s in range(0, p, chunk):
        delta = x[s:s + chunk].to(device()) - mean[s:s + chunk, None].to(device())
        sq = (torch.bmm(delta, inv[s:s + chunk].to(device())) * delta).sum(dim=-1)
        out[s:s + chunk] = sq.clamp_min(0).sqrt().cpu()
    return out


class PadimMethod(AnomalyMethod):
    name = "padim"
    produces_maps = True

    def __init__(self, backbone: str = "resnet18", layers=("layer1", "layer2", "layer3"),
                 n_features: int | None = None, epsilon: float = 0.01, sigma: float = 4.0,
                 batch_size: int = 32, chunk: int = 256, pretrained: bool = True, seed: int = 0):
        timm_name(backbone)
        self.backbone = backbone
        self.layers = list(layers)
        self.n_features = PADIM_FEATURES[backbone] if n_features is None else int(n_features)
        self.epsilon = epsilon
        self.sigma = sigma
        self.batch_size = batch_size
        self.chunk = chunk
        self.pretrained = pretrained
        self.seed = seed

    @torch.no_grad()
    def _embed(self, images: np.ndarray) -> torch.Tensor:
        model = load_backbone(self.backbone, self.layers, self.pretrained)
        index = self.index_.to(device())
        out = []
        for x in batches(images, self.batch_size):
            maps = model(x)
            size = maps[0].shape[-2:]
            emb = torch.cat([maps[0]] + [F.interpolate(m, size=size, mode="nearest") for m in maps[1:]], dim=1)
            out.append(emb[:, index].cpu())
        emb = torch.cat(out)
        self.grid_ = tuple(emb.shape[-2:])
        return emb.flatten(2).permute(2, 0, 1)

    def fit(self, images: np.ndarray) -> "PadimMethod":
        total = sum(load_backbone(self.backbone, self.layers, self.pretrained).feature_info.channels())
        if not 0 < self.n_features <= total:
            raise ValueError(f"n_features must be in 1..{total}, got {self.n_features}")
        self.embedding_dim_ = int(total)
        gen = torch.Generator().manual_seed(self.seed)
        self.index_ = torch.randperm(total, generator=gen)[:self.n_features].sort().values
        self.mean_, self.inv_cov_ = fit_gaussians(self._embed(images), self.epsilon, self.chunk)
        return self

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> Prediction:
        dist = mahalanobis(self._embed(images), self.mean_, self.inv_cov_, self.chunk)
        maps = dist.T.reshape(len(images), 1, *self.grid_)
        maps = F.interpolate(maps, size=images.shape[1:3], mode="bilinear", align_corners=False)
        maps = gaussian_blur(maps, self.sigma)[:, 0]
        return Prediction(scores=maps.amax(dim=(1, 2)).numpy(), maps=maps.numpy())

    def model_size(self) -> int:
        return int(self.mean_.numel() + self.inv_cov_.numel() + self.index_.numel())

    def get_config(self) -> dict:
        cfg = {"backbone": self.backbone, "weights": timm_name(self.backbone), "layers": self.layers,
               "n_features": self.n_features, "epsilon": self.epsilon, "sigma": self.sigma,
               "pretrained": self.pretrained, "seed": self.seed,
               "backbone_params": backbone_params(self.backbone, self.layers, self.pretrained)}
        if hasattr(self, "embedding_dim_"):
            cfg.update(embedding_dim=self.embedding_dim_, grid=list(self.grid_))
        return cfg


def original_embedding(features: dict, layers, embed_dim: int) -> torch.Tensor:
    base = features[layers[0]]
    b, _, h, w = base.shape
    out = []
    for layer in layers:
        f = features[layer]
        if f.shape[-2:] != base.shape[-2:]:
            f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        out.append(F.adaptive_avg_pool1d(f.permute(0, 2, 3, 1).reshape(-1, 1, f.shape[1]), embed_dim))
    emb = F.adaptive_avg_pool1d(torch.cat(out, dim=2), embed_dim)
    return emb.reshape(b, h, w, embed_dim).permute(0, 3, 1, 2)


class PatchcoreMethod(AnomalyMethod):
    name = "patchcore"
    produces_maps = True

    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"),
                 coreset_ratio: float = 0.1, num_neighbors: int = 9, aggregation: str = "concat",
                 embed_dim: int = 1024, batch_size: int = 32, pretrained: bool = True, seed: int = 0):
        timm_name(backbone)
        if not 0 < coreset_ratio <= 1:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown aggregation: {aggregation}")
        self.aggregation = aggregation
        self.embed_dim = int(embed_dim)
        self.backbone = backbone
        self.layers = list(layers)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.seed = seed

    def fit(self, images: np.ndarray) -> "PatchcoreMethod":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        self.model_ = PatchcoreModel(layers=list(self.layers), backbone=timm_name(self.backbone),
                                     pre_trained=self.pretrained, num_neighbors=self.num_neighbors)
        if self.aggregation == "original":
            self.model_.generate_embedding = lambda f: original_embedding(f, self.layers, self.embed_dim)
        self.model_.to(device()).train()
        with torch.no_grad():
            for x in batches(images, self.batch_size):
                self.model_(x)
            self.n_patches_ = int(sum(len(e) for e in self.model_.embedding_store))
            self.model_.subsample_embedding(self.coreset_ratio)
        self.model_.eval()
        return self

    @torch.no_grad()
    def predict(self, images: np.ndarray) -> Prediction:
        scores, maps = [], []
        for x in batches(images, self.batch_size):
            out = self.model_(x)
            scores.append(out.pred_score.reshape(-1).cpu())
            maps.append(out.anomaly_map[:, 0].cpu())
        return Prediction(scores=torch.cat(scores).numpy(), maps=torch.cat(maps).numpy())

    def model_size(self) -> int:
        return int(self.model_.memory_bank.numel())

    def get_config(self) -> dict:
        cfg = {"backbone": self.backbone, "weights": timm_name(self.backbone), "layers": self.layers,
               "coreset_ratio": self.coreset_ratio, "num_neighbors": self.num_neighbors,
               "aggregation": self.aggregation, "pretrained": self.pretrained, "seed": self.seed,
               "backbone_params": backbone_params(self.backbone, self.layers, self.pretrained)}
        if self.aggregation == "original":
            cfg["embed_dim"] = self.embed_dim
        if hasattr(self, "model_"):
            cfg.update(n_patches=self.n_patches_, memory_bank=list(self.model_.memory_bank.shape))
        return cfg
