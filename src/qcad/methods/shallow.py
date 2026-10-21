from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .. import config
from .base import AnomalyMethod, Prediction

FEATURES = ("hog", "lbp", "glcm", "color")
DETECTORS = ("ocsvm", "iforest", "gmm", "mahalanobis")
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])

_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 64


def to_gray(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image[..., :3] @ GRAY_WEIGHTS), 0, 255).astype(np.uint8)


def _regions(image: np.ndarray, grid: int):
    h, w = image.shape[:2]
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            yield image[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]


def hog_features(image: np.ndarray, cell: int = 16, orientations: int = 9) -> np.ndarray:
    return hog(to_gray(image), orientations=orientations, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)


def lbp_features(image: np.ndarray, scales=((8, 1), (16, 2)), grid: int = 4) -> np.ndarray:
    gray = to_gray(image)
    out = []
    for p, r in scales:
        codes = local_binary_pattern(gray, int(p), float(r), method="uniform").astype(int)
        for region in _regions(codes, grid):
            out.append(np.bincount(region.ravel(), minlength=int(p) + 2) / region.size)
    return np.concatenate(out)


def glcm_features(image: np.ndarray, levels: int = 32, distances=(1, 3), grid: int = 4) -> np.ndarray:
    quant = (to_gray(image).astype(np.int32) * levels // 256).astype(np.uint8)
    out = []
    for region in _regions(quant, grid):
        m = graycomatrix(region, list(distances), list(GLCM_ANGLES), levels=levels,
                         symmetric=True, normed=True)
        out.extend(graycoprops(m, prop).ravel() for prop in GLCM_PROPS)
    return np.nan_to_num(np.concatenate(out), nan=1.0)


def color_features(image: np.ndarray, bins: int = 16, grid: int = 4) -> np.ndarray:
    out = []
    for region in _regions(image, grid):
        for c in range(3):
            values = region[..., c].ravel().astype(int) * bins // 256
            out.append(np.bincount(values, minlength=bins) / values.size)
    return np.concatenate(out)


EXTRACTORS = {"hog": hog_features, "lbp": lbp_features, "glcm": glcm_features, "color": color_features}


def _extract_chunk(fn, images, kwargs) -> np.ndarray:
    return np.stack([fn(img, **kwargs) for img in images]).astype(np.float32)


def _compute_block(images: np.ndarray, name: str, n_jobs: int, kwargs: dict) -> np.ndarray:
    fn = EXTRACTORS[name]
    if n_jobs == 1 or len(images) < 16:
        return _extract_chunk(fn, images, kwargs)
    chunks = [c for c in np.array_split(images, min(len(images), 32)) if len(c)]
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(_extract_chunk)(fn, c, kwargs) for c in chunks))


def extract_block(images: np.ndarray, name: str, n_jobs: int = -1, use_cache: bool = True,
                  **kwargs) -> np.ndarray:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown feature: {name}")
    images = np.ascontiguousarray(images)
    if not use_cache:
        return _compute_block(images, name, n_jobs, kwargs)
    params = repr(sorted((k, repr(v)) for k, v in kwargs.items()))
    digest = hashlib.sha1(images.tobytes() + repr(images.shape).encode() + params.encode()).hexdigest()
    key = (name, digest)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    path = config.CACHE_DIR / "features" / f"{name}_{digest}.npy"
    if path.exists():
        feats = np.load(path)
    else:
        feats = _compute_block(images, name, n_jobs, kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, feats)
    _CACHE[key] = feats
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return feats


class BlockReducer:
    def __init__(self, n_components: int | None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, x: np.ndarray) -> "BlockReducer":
        self.scaler_ = StandardScaler().fit(x)
        z = self.scaler_.transform(x)
        self.pca_ = None
        if self.n_components is not None:
            k = max(1, min(int(self.n_components), len(x) - 1, x.shape[1]))
            self.pca_ = PCA(n_components=k, whiten=self.whiten, svd_solver="full").fit(z)
            z = self.pca_.transform(z)
        total = float(np.sum(np.var(z, axis=0)))
        self.scale_ = 1.0 / np.sqrt(total) if total > 0 else 1.0
        self.n_out_ = z.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler_.transform(x)
        if self.pca_ is not None:
            z = self.pca_.transform(z)
        return z * self.scale_

    def n_values(self) -> int:
        n = self.scaler_.mean_.size + self.scaler_.scale_.size + 1
        if self.pca_ is not None:
            n += self.pca_.components_.size + self.pca_.mean_.size
            if self.whiten:
                n += self.pca_.explained_variance_.size
        return int(n)


class ShallowMethod(AnomalyMethod):
    name = "shallow"
    produces_maps = False

    def __init__(self, features=FEATURES, detector: str = "ocsvm",
                 pca_components: int | None = 20, whiten: bool = False,
                 hog_cell: int = 16, hog_orientations: int = 9,
                 lbp_scales=((8, 1), (16, 2)), grid: int = 4,
                 glcm_levels: int = 32, glcm_distances=(1, 3), color_bins: int = 16,
                 nu: float = 0.1, gamma_factor: float = 1.0,
                 n_estimators: int = 100, max_samples: str | int = "auto",
                 n_components: int = 1, covariance_type: str = "full", reg_covar: float = 1e-3,
                 covariance: str = "ledoitwolf", n_jobs: int = -1, feature_cache: bool = True,
                 seed: int = 0):
        features = [features] if isinstance(features, str) else list(features)
        unknown = [f for f in features if f not in FEATURES]
        if unknown or not features:
            raise ValueError(f"unknown features: {unknown}")
        if detector not in DETECTORS:
            raise ValueError(f"unknown detector: {detector}")
        if covariance not in ("ledoitwolf", "empirical"):
            raise ValueError(f"unknown covariance: {covariance}")
        self.features = features
        self.detector = detector
        self.pca_components = pca_components
        self.whiten = whiten
        self.hog_cell = hog_cell
        self.hog_orientations = hog_orientations
        self.lbp_scales = [tuple(int(v) for v in s) for s in lbp_scales]
        self.grid = grid
        self.glcm_levels = glcm_levels
        self.glcm_distances = [int(d) for d in glcm_distances]
        self.color_bins = color_bins
        self.nu = nu
        self.gamma_factor = gamma_factor
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.covariance = covariance
        self.n_jobs = n_jobs
        self.feature_cache = feature_cache
        self.seed = seed

    def _block_kwargs(self, name: str) -> dict:
        return {
            "hog": {"cell": self.hog_cell, "orientations": self.hog_orientations},
            "lbp": {"scales": tuple(self.lbp_scales), "grid": self.grid},
            "glcm": {"levels": self.glcm_levels, "distances": tuple(self.glcm_distances), "grid": self.grid},
            "color": {"bins": self.color_bins, "grid": self.grid},
        }[name]

    def extract(self, images: np.ndarray) -> list[np.ndarray]:
        return [extract_block(images, f, n_jobs=self.n_jobs, use_cache=self.feature_cache, **self._block_kwargs(f))
                for f in self.features]

    def transform(self, images: np.ndarray) -> np.ndarray:
        return np.concatenate([r.transform(b) for r, b in zip(self.reducers_, self.extract(images))], axis=1)

    def _make_detector(self, x: np.ndarray):
        n_samples = len(x)
        if self.detector == "ocsvm":
            var = float(x.var())
            self.gamma_ = self.gamma_factor / (x.shape[1] * var) if var > 0 else self.gamma_factor
            return OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma_)
        if self.detector == "iforest":
            return IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                   random_state=self.seed)
        if self.detector == "gmm":
            return GaussianMixture(n_components=min(self.n_components, n_samples),
                                   covariance_type=self.covariance_type, reg_covar=self.reg_covar,
                                   random_state=self.seed)
        return LedoitWolf() if self.covariance == "ledoitwolf" else EmpiricalCovariance()

    def fit(self, images: np.ndarray) -> "ShallowMethod":
        blocks = self.extract(images)
        self.reducers_ = [BlockReducer(self.pca_components, self.whiten).fit(b) for b in blocks]
        x = np.concatenate([r.transform(b) for r, b in zip(self.reducers_, blocks)], axis=1)
        self.block_dims_ = {f: [int(b.shape[1]), int(r.n_out_)]
                            for f, b, r in zip(self.features, blocks, self.reducers_)}
        self.model_ = self._make_detector(x).fit(x)
        return self

    def predict(self, images: np.ndarray) -> Prediction:
        z = self.transform(images)
        if self.detector == "ocsvm":
            scores = -self.model_.decision_function(z)
        elif self.detector in ("iforest", "gmm"):
            scores = -self.model_.score_samples(z)
        else:
            scores = self.model_.mahalanobis(z)
        return Prediction(scores=scores)

    def _detector_size(self) -> int:
        m = self.model_
        if self.detector == "ocsvm":
            return m.support_vectors_.size + m.dual_coef_.size + m.intercept_.size
        if self.detector == "iforest":
            return 5 * sum(t.tree_.node_count for t in m.estimators_)
        if self.detector == "gmm":
            return m.weights_.size + m.means_.size + m.precisions_cholesky_.size
        return m.location_.size + m.precision_.size

    def model_size(self) -> int:
        return int(sum(r.n_values() for r in self.reducers_) + self._detector_size())

    def get_config(self) -> dict:
        cfg = {
            "features": self.features, "detector": self.detector,
            "pca_components": self.pca_components, "whiten": self.whiten,
            "hog_cell": self.hog_cell, "hog_orientations": self.hog_orientations,
            "lbp_scales": self.lbp_scales, "grid": self.grid,
            "glcm_levels": self.glcm_levels, "glcm_distances": self.glcm_distances,
            "color_bins": self.color_bins, "feature_cache": self.feature_cache, "seed": self.seed,
        }
        cfg.update({
            "ocsvm": {"nu": self.nu, "gamma_factor": self.gamma_factor},
            "iforest": {"n_estimators": self.n_estimators, "max_samples": self.max_samples},
            "gmm": {"n_components": self.n_components, "covariance_type": self.covariance_type,
                    "reg_covar": self.reg_covar},
            "mahalanobis": {"covariance": self.covariance},
        }[self.detector])
        if hasattr(self, "block_dims_"):
            cfg["block_dims"] = self.block_dims_
        if hasattr(self, "gamma_"):
            cfg["gamma"] = self.gamma_
        return cfg
