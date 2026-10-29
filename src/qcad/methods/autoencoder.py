from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import config
from .base import AnomalyMethod, Prediction

LOSSES = ("mse", "ssim")
SCORE_REDUCTIONS = ("max", "mean", "top")
GRAY_WEIGHTS = (0.299, 0.587, 0.114)
CACHE_VERSION = 1

AUGMENTATIONS = {
    "carpet": {"hflip": True, "vflip": True, "rot90": True, "rotate": 0.0, "shift": 0},
    "grid": {"hflip": True, "vflip": True, "rot90": True, "rotate": 0.0, "shift": 0},
    "leather": {"hflip": True, "vflip": True, "rot90": True, "rotate": 0.0, "shift": 0},
    "tile": {"hflip": True, "vflip": True, "rot90": True, "rotate": 0.0, "shift": 0},
    "wood": {"hflip": True, "vflip": True, "rot90": False, "rotate": 0.0, "shift": 0},
    "bottle": {"hflip": True, "vflip": True, "rot90": False, "rotate": 180.0, "shift": 8},
    "cable": {"hflip": False, "vflip": False, "rot90": False, "rotate": 5.0, "shift": 8},
    "capsule": {"hflip": False, "vflip": False, "rot90": False, "rotate": 3.0, "shift": 8},
    "hazelnut": {"hflip": True, "vflip": True, "rot90": False, "rotate": 180.0, "shift": 8},
    "metal_nut": {"hflip": False, "vflip": False, "rot90": False, "rotate": 180.0, "shift": 8},
    "pill": {"hflip": False, "vflip": False, "rot90": False, "rotate": 3.0, "shift": 8},
    "screw": {"hflip": False, "vflip": False, "rot90": False, "rotate": 180.0, "shift": 8},
    "toothbrush": {"hflip": True, "vflip": False, "rot90": False, "rotate": 3.0, "shift": 8},
    "transistor": {"hflip": False, "vflip": False, "rot90": False, "rotate": 0.0, "shift": 0},
    "zipper": {"hflip": True, "vflip": True, "rot90": False, "rotate": 0.0, "shift": 8},
}
NO_AUGMENTATION = {"hflip": False, "vflip": False, "rot90": False, "rotate": 0.0, "shift": 0}


def gaussian_kernel(sigma: float, size: int | None = None, device=None) -> torch.Tensor:
    size = size or 2 * math.ceil(4 * sigma) + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _separable(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c, pad = x.shape[1], len(kernel) // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    return _separable(x, gaussian_kernel(sigma, device=x.device))


def ssim_map(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5,
             k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, window, device=x.device)
    c1, c2 = k1 ** 2, k2 ** 2
    mx, my = _separable(x, kernel), _separable(y, kernel)
    sxx = _separable(x * x, kernel) - mx ** 2
    syy = _separable(y * y, kernel) - my ** 2
    sxy = _separable(x * y, kernel) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))


def residual_map(x: torch.Tensor, recon: torch.Tensor, loss: str, window: int = 11) -> torch.Tensor:
    if loss == "mse":
        return ((x - recon) ** 2).mean(dim=1)
    return (1 - ssim_map(x, recon, window)).mean(dim=1)


def _down(c_in, c_out, k, s, p):
    return [nn.Conv2d(c_in, c_out, k, s, p), nn.LeakyReLU(0.2)]


def _up(c_in, c_out, k, s, p):
    return [nn.ConvTranspose2d(c_in, c_out, k, s, p), nn.LeakyReLU(0.2)]


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size: int, channels: int = 3, latent_dim: int = 100):
        super().__init__()
        extra = input_size > 128
        bottom = input_size // (32 if extra else 16)
        if bottom * (32 if extra else 16) != input_size:
            raise ValueError(f"unsupported input size: {input_size}")
        enc = [*_down(channels, 32, 4, 2, 1), *_down(32, 32, 4, 2, 1), *_down(32, 32, 3, 1, 1),
               *_down(32, 64, 4, 2, 1), *_down(64, 64, 3, 1, 1), *_down(64, 128, 4, 2, 1)]
        if extra:
            enc += _down(128, 128, 4, 2, 1)
        enc += [*_down(128, 64, 3, 1, 1), *_down(64, 32, 3, 1, 1), nn.Conv2d(32, latent_dim, bottom, 1, 0)]
        dec = [*_up(latent_dim, 32, bottom, 1, 0), *_down(32, 64, 3, 1, 1), *_down(64, 128, 3, 1, 1)]
        if extra:
            dec += _up(128, 128, 4, 2, 1)
        dec += [*_up(128, 64, 4, 2, 1), *_down(64, 64, 3, 1, 1), *_up(64, 32, 4, 2, 1),
                *_down(32, 32, 3, 1, 1), *_up(32, 32, 4, 2, 1), nn.ConvTranspose2d(32, channels, 4, 2, 1)]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def augment(x: torch.Tensor, spec: dict, gen: torch.Generator) -> torch.Tensor:
    n, dev = len(x), x.device

    def coin():
        return (torch.rand(n, generator=gen, device=dev) < 0.5).view(-1, 1, 1, 1)

    if spec["hflip"]:
        x = torch.where(coin(), x.flip(-1), x)
    if spec["vflip"]:
        x = torch.where(coin(), x.flip(-2), x)
    if spec["rot90"]:
        k = torch.randint(0, 4, (n,), generator=gen, device=dev)
        x = torch.stack([torch.rot90(img, int(r), dims=(-2, -1)) for img, r in zip(x, k)])
    if spec["rotate"] > 0 or spec["shift"] > 0:
        angle = (torch.rand(n, generator=gen, device=dev) * 2 - 1) * math.radians(spec["rotate"])
        shift = (torch.rand(n, 2, generator=gen, device=dev) * 2 - 1) * 2 * spec["shift"] / x.shape[-1]
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                             torch.stack([sin, cos, shift[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return x


def random_crops(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    h, w = x.shape[-2:]
    if size >= h and size >= w:
        return x
    ys = torch.randint(0, h - size + 1, (len(x),), generator=gen, device=x.device).tolist()
    xs = torch.randint(0, w - size + 1, (len(x),), generator=gen, device=x.device).tolist()
    return torch.stack([img[:, y:y + size, c:c + size] for img, y, c in zip(x, ys, xs)])


def patch_positions(length: int, size: int, stride: int) -> list[int]:
    pos = list(range(0, length - size + 1, stride))
    if pos[-1] != length - size:
        pos.append(length - size)
    return pos


class AutoencoderMethod(AnomalyMethod):
    name = "ae"
    produces_maps = True

    def __init__(self, loss: str = "mse", latent_dim: int = 100, grayscale: bool | None = None,
                 texture_patch: int | None = 128, object_patch: int | None = None, patch_stride: int = 32,
                 augmentation: bool = True, max_epochs: int = 100, samples_per_epoch: int = 10000,
                 batch_size: int = 128, lr: float = 2e-4, weight_decay: float = 1e-5,
                 patience: int = 10, mse_warmup: int = 0, ssim_window: int = 11, map_sigma: float = 4.0,
                 score: str = "max", score_top: float = 0.01, eval_batch: int = 32,
                 device: str = "auto", model_cache: bool = True, category: str | None = None,
                 seed: int = 0):
        if loss not in LOSSES:
            raise ValueError(f"unknown loss: {loss}")
        if score not in SCORE_REDUCTIONS:
            raise ValueError(f"unknown score: {score}")
        if category is not None and category not in AUGMENTATIONS:
            raise ValueError(f"unknown category: {category}")
        self.loss = loss
        self.latent_dim = latent_dim
        self.grayscale = loss == "ssim" if grayscale is None else bool(grayscale)
        self.texture_patch = texture_patch
        self.object_patch = object_patch
        self.patch_stride = patch_stride
        self.augmentation = augmentation
        self.max_epochs = max_epochs
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.mse_warmup = mse_warmup
        self.ssim_window = ssim_window
        self.map_sigma = map_sigma
        self.score = score
        self.score_top = score_top
        self.eval_batch = eval_batch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        self.model_cache = model_cache
        self.category = category
        self.seed = seed

    @property
    def patch_size(self) -> int | None:
        is_texture = self.category is not None and config.category_type(self.category) == "texture"
        return self.texture_patch if is_texture else self.object_patch

    @property
    def augment_spec(self) -> dict:
        if not self.augmentation or self.category is None:
            return NO_AUGMENTATION
        return AUGMENTATIONS[self.category]

    def _prepare(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(images).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        if self.grayscale:
            w = torch.tensor(GRAY_WEIGHTS, device=self.device).view(1, 3, 1, 1)
            x = (x * w).sum(dim=1, keepdim=True)
        return x

    def _loss(self, x: torch.Tensor, recon: torch.Tensor, loss: str | None = None) -> torch.Tensor:
        if (loss or self.loss) == "mse":
            return F.mse_loss(recon, x)
        return 1 - ssim_map(x, recon, self.ssim_window).mean()

    def _train_params(self) -> dict:
        return {"loss": self.loss, "latent_dim": self.latent_dim, "grayscale": self.grayscale,
                "patch_size": self.patch_size, "augment": self.augment_spec,
                "max_epochs": self.max_epochs, "samples_per_epoch": self.samples_per_epoch,
                "batch_size": self.batch_size, "lr": self.lr, "weight_decay": self.weight_decay,
                "patience": self.patience, "mse_warmup": self.mse_warmup, "ssim_window": self.ssim_window, "seed": self.seed,
                "patch_stride": self.patch_stride, "version": CACHE_VERSION}

    def _cache_file(self, images: np.ndarray, holdout: np.ndarray | None):
        h = hashlib.sha1(json.dumps(self._train_params(), sort_keys=True).encode())
        for arr in (images, holdout):
            if arr is not None:
                arr = np.ascontiguousarray(arr)
                h.update(repr(arr.shape).encode() + arr.tobytes())
        return config.CACHE_DIR / "ae" / f"{h.hexdigest()}.pt"

    def _build(self, input_size: int) -> ConvAutoencoder:
        return ConvAutoencoder(input_size, 1 if self.grayscale else 3, self.latent_dim).to(self.device)

    def fit(self, images: np.ndarray, holdout: np.ndarray | None = None) -> "AutoencoderMethod":
        size = self.patch_size or images.shape[1]
        path = self._cache_file(images, holdout) if self.model_cache else None
        if path is not None and path.exists():
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model_ = self._build(size)
            self.model_.load_state_dict(state["model"])
            self.history_ = state["history"]
            self.model_.eval()
            return self
        torch.manual_seed(self.seed)
        self.model_ = self._build(size)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        data = torch.as_tensor(np.ascontiguousarray(images)).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = math.ceil(self.samples_per_epoch / self.batch_size)
        best, best_state, wait = math.inf, None, 0
        history = {"train_loss": [], "holdout_loss": [], "best_epoch": None}
        for epoch in range(self.max_epochs):
            self.model_.train()
            total = 0.0
            for _ in range(steps):
                idx = torch.randint(0, len(data), (self.batch_size,), generator=gen, device=self.device)
                x = augment(self._prepare(data[idx]), self.augment_spec, gen)
                if self.patch_size:
                    x = random_crops(x, self.patch_size, gen)
                loss = self._loss(x, self.model_(x), "mse" if epoch < self.mse_warmup else None)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
            history["train_loss"].append(total / steps)
            if holdout is None:
                history["best_epoch"] = epoch
                continue
            val = self._holdout_loss(holdout)
            history["holdout_loss"].append(val)
            if epoch < self.mse_warmup:
                continue
            if val < best:
                best, wait, history["best_epoch"] = val, 0, epoch
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.history_ = history
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save({"model": self.model_.state_dict(), "history": history}, tmp)
            tmp.replace(path)
        return self

    @torch.no_grad()
    def _holdout_loss(self, holdout: np.ndarray) -> float:
        self.model_.eval()
        losses, n = 0.0, 0
        for i in range(0, len(holdout), self.eval_batch):
            x = self._prepare(holdout[i:i + self.eval_batch])
            losses += self._loss(x, self.reconstruct(x)).item() * len(x)
            n += len(x)
        return losses / n

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        if not p or p >= x.shape[-1]:
            return self.model_(x)
        h, w = x.shape[-2:]
        out, count = torch.zeros_like(x), torch.zeros_like(x[:1, :1])
        for y in patch_positions(h, p, self.patch_stride):
            for c in patch_positions(w, p, self.patch_stride):
                out[..., y:y + p, c:c + p] += self.model_(x[..., y:y + p, c:c + p])
                count[..., y:y + p, c:c + p] += 1
        return out / count

    @torch.no_grad()
    def anomaly_maps(self, images: np.ndarray) -> np.ndarray:
        maps = []
        for i in range(0, len(images), self.eval_batch):
            x = self._prepare(images[i:i + self.eval_batch])
            m = residual_map(x, self.reconstruct(x), self.loss, self.ssim_window)
            maps.append(gaussian_blur(m[:, None], self.map_sigma)[:, 0].cpu())
        return torch.cat(maps).numpy().astype(np.float32)

    def reduce_scores(self, maps: np.ndarray) -> np.ndarray:
        flat = maps.reshape(len(maps), -1)
        if self.score == "max":
            return flat.max(axis=1)
        if self.score == "mean":
            return flat.mean(axis=1)
        k = max(1, int(round(self.score_top * flat.shape[1])))
        return np.sort(flat, axis=1)[:, -k:].mean(axis=1)

    def predict(self, images: np.ndarray) -> Prediction:
        maps = self.anomaly_maps(images)
        return Prediction(scores=self.reduce_scores(maps), maps=maps)

    def model_size(self) -> int:
        return int(sum(p.numel() for p in self.model_.parameters()))

    def fit_info(self) -> dict:
        if not hasattr(self, "history_"):
            return {}
        holdout = self.history_["holdout_loss"]
        best = self.history_["best_epoch"]
        return {"epochs_run": len(self.history_["train_loss"]), "best_epoch": best,
                "holdout_loss": holdout[best] if holdout else np.nan}

    def get_config(self) -> dict:
        return {**self._train_params(), "texture_patch": self.texture_patch, "object_patch": self.object_patch,
                "map_sigma": self.map_sigma, "score": self.score, "score_top": self.score_top,
                "model_cache": self.model_cache, "device": str(self.device)}
