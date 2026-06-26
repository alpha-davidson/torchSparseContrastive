"""
3D point cloud augmentations for SimCLR contrastive learning.

Each augmentation takes a numpy array of shape (N, 3) — xyz coordinates —
and returns an augmented copy of the same shape.

Typical usage
-------------
    aug = Compose([
        RandomRotation(axes='y'),
        RandomJitter(sigma=0.01, clip=0.05),
        RandomScale(lo=0.8, hi=1.25),
        RandomPointDropout(p=0.1),
    ])
    view1 = aug(pts)
    view2 = aug(pts)
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Augmentation:
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

class RandomRotation(Augmentation):
    """
    Rotate the point cloud by a random angle around one or more axes.

    Parameters
    ----------
    axes : str
        Any combination of 'x', 'y', 'z'. 'y' alone is the most common
        for ShapeNet (upright objects). Use 'xyz' for SO(3) rotation.
    angle_range : tuple
        (min_deg, max_deg) rotation range. Default is full 360°.
    """

    def __init__(self, axes: str = "y", angle_range: tuple = (0, 360)):
        self.axes        = axes
        self.angle_range = angle_range

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        lo, hi = np.radians(self.angle_range[0]), np.radians(self.angle_range[1])
        for axis in self.axes:
            theta = np.random.uniform(lo, hi)
            c, s  = np.cos(theta), np.sin(theta)
            if axis == "x":
                R = np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
            elif axis == "y":
                R = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
            else:  # z
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            pts = pts @ R.T
        return pts


class RandomJitter(Augmentation):
    """
    Add Gaussian noise to each point independently.

    Parameters
    ----------
    sigma : float
        Standard deviation of the noise.
    clip  : float
        Maximum absolute noise value (clips outliers).
    """

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip  = clip

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        noise = np.clip(
            np.random.normal(0, self.sigma, pts.shape).astype(np.float32),
            -self.clip, self.clip
        )
        return pts + noise


class RandomScale(Augmentation):
    """
    Uniformly scale the point cloud by a random factor.

    Parameters
    ----------
    lo, hi : float
        Range of the scale factor. Default 0.8–1.25.
    """

    def __init__(self, lo: float = 0.8, hi: float = 1.25):
        self.lo = lo
        self.hi = hi

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.lo, self.hi)
        return pts * scale


class RandomFlip(Augmentation):
    """
    Randomly flip the point cloud along one axis with probability p.

    Parameters
    ----------
    axis : int
        0=x, 1=y, 2=z. Default is x-axis (left-right flip).
    p    : float
        Probability of applying the flip.
    """

    def __init__(self, axis: int = 0, p: float = 0.5):
        self.axis = axis
        self.p    = p

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            pts = pts.copy()
            pts[:, self.axis] *= -1
        return pts


class RandomPointDropout(Augmentation):
    """
    Randomly drop a fraction of points and sample replacements.

    This simulates occlusion and sensor dropout. The output always has
    the same number of points as the input (dropped points are replaced
    by resampling from the remaining points).

    Parameters
    ----------
    p : float
        Fraction of points to drop (0–1). Default 0.1 = drop 10%.
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        N        = len(pts)
        keep_n   = max(1, int(N * (1 - self.p)))
        keep_idx = np.random.choice(N, keep_n, replace=False)
        kept     = pts[keep_idx]
        # resample to restore original count
        fill_idx = np.random.choice(keep_n, N - keep_n, replace=True)
        return np.concatenate([kept, kept[fill_idx]], axis=0)


class RandomShift(Augmentation):
    """
    Randomly translate the point cloud.

    Parameters
    ----------
    max_shift : float
        Maximum shift in each axis. Default 0.1.
    """

    def __init__(self, max_shift: float = 0.1):
        self.max_shift = max_shift

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(-self.max_shift, self.max_shift, (1, 3)).astype(np.float32)
        return pts + shift


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class Compose(Augmentation):
    """Apply a list of augmentations sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            pts = t(pts)
        return pts


# ---------------------------------------------------------------------------
# Preset augmentation pipelines
# ---------------------------------------------------------------------------

def simclr_augmentation(
    rotation_axes: str  = "y",
    jitter_sigma: float = 0.01,
    jitter_clip: float  = 0.05,
    scale_lo: float     = 0.8,
    scale_hi: float     = 1.25,
    dropout_p: float    = 0.1,
    flip_p: float       = 0.5,
) -> Compose:
    """
    Standard SimCLR augmentation pipeline for 3D point clouds.

    Two independent applications of this pipeline to the same point cloud
    produce view1 and view2 for contrastive learning.

    Default settings are conservative — suitable for ShapeNet and AT-TPC.
    For AT-TPC data you may want to increase jitter_sigma and dropout_p
    to simulate detector noise and missing hits.
    """
    return Compose([
        RandomRotation(axes=rotation_axes),
        RandomScale(lo=scale_lo, hi=scale_hi),
        RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
        RandomFlip(axis=0, p=flip_p),
        RandomPointDropout(p=dropout_p),
        RandomShift(max_shift=0.1),
    ])

##looked up realistic augmentations for AT-TPC
def attpc_augmentation() -> Compose:
    """
    Augmentation pipeline tuned for AT-TPC particle track data.

    Stronger jitter and dropout to simulate detector noise,
    missing pad hits, and gain variations. No flip since track
    direction is physically meaningful.
    """
    return Compose([
        RandomRotation(axes="y", angle_range=(0, 360)),
        RandomScale(lo=0.9, hi=1.1),
        RandomJitter(sigma=0.02, clip=0.1),
        RandomPointDropout(p=0.2),
        RandomShift(max_shift=0.05),
    ])
