"""Stain SAN SVD-compatible baseline and experimental samplers.

Based on Kim et al., DOI 10.1117/1.JMI.11.4.044006, and their MIT code.
See upstream/LICENSE and README.md for attribution and implementation differences.
"""
import numpy as np


def rgb2od(image):
    return -np.log(np.clip(np.asarray(image, dtype=float), 1e-4, 255) / 255)


def extract(image):
    """Return a 3x2 stain basis; fail explicitly on unsuitable tissue images."""
    od = rgb2od(image).reshape(-1, 3)
    tissue = od[np.sum(od ** 2, axis=1) > 0.3 ** 2]
    if len(tissue) < 10 or np.std(tissue) < 1e-8:
        raise ValueError("Insufficient nonconstant tissue for SVD extraction")
    values, vectors = np.linalg.eigh(np.cov(tissue.T))
    if values[-2] < 1e-10:
        raise ValueError("Degenerate stain plane")
    plane = vectors[:, 1:3]
    coordinates = tissue @ plane
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    low, high = np.percentile(angles, [1, 99])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    basis = np.stack([first, second] if first[0] > second[0] else [second, first], axis=1)
    basis = np.clip(basis, 0, None)
    if np.linalg.matrix_rank(basis) < 2:
        raise ValueError("Degenerate stain basis")
    return basis


def fit(bases, shrinkage=0.1):
    if len(bases) < 10:
        raise ValueError("Need at least 10 valid training stain matrices")
    values = np.stack(bases).reshape(len(bases), 6)
    center = np.median(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=0)
    variance = np.trace(covariance) / 6
    # Shrinkage stabilizes the covariance while keeping the same total variance.
    elliptical = (1 - shrinkage) * covariance + shrinkage * variance * np.eye(6)
    return {"center": center, "covariance": elliptical, "variance": variance,
            "shrinkage": shrinkage, "fit_images": len(bases)}


def sample(parameters, mode, rng):
    mean = np.asarray(parameters["center"])
    variance = float(parameters["variance"])
    if mode == "spherical":
        draw = mean + rng.normal(0, np.sqrt(variance), 6)
    elif mode == "elliptical":
        draw = rng.multivariate_normal(mean, np.asarray(parameters["covariance"]))
    elif mode == "uniform":
        # Independent uniform components have the same pre-clipping variance.
        half_width = np.sqrt(3 * variance)
        draw = mean + rng.uniform(-half_width, half_width, 6)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.clip(draw, 0, None).reshape(3, 2)


def reconstruct(image, source_basis, target_basis, intensity_scale=1.0):
    od = rgb2od(image).reshape(-1, 3)
    concentrations = np.linalg.lstsq(source_basis, od.T, rcond=None)[0]
    updated = (target_basis @ (concentrations * intensity_scale)).T
    background = np.sum(od ** 2, axis=1) <= 0.3 ** 2
    updated[background] = od[background]
    rgb = np.clip(255 * np.exp(-np.clip(updated, 1e-4, None)), 0, 255)
    return rgb.reshape(image.shape).astype(np.uint8)


def transform(image, parameters, mode, rng, training=True):
    try:
        source = extract(image)
    except ValueError:
        return image.copy(), True
    target = sample(parameters, mode, rng) if training else np.asarray(parameters["center"]).reshape(3, 2)
    scale = 1 + rng.uniform(-0.2, 0.2) if training else 1.0
    return reconstruct(image, source, target, scale), False
