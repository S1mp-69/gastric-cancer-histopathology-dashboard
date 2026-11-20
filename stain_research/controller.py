"""Experimental controller; diagnostics are numerical proxies, not pathology quality."""
import numpy as np

from transforms import extract, rgb2od, sample


def diagnostics(image):
    basis = extract(image)
    od = rgb2od(image).reshape(-1, 3)
    tissue = (od ** 2).sum(axis=1) > 0.09
    concentrations = np.linalg.lstsq(basis, od.T, rcond=None)[0]
    residual = np.linalg.norm((od.T - basis @ concentrations)[:, tissue])
    residual /= max(np.linalg.norm(od[tissue]), 1e-12)
    return np.array([np.log(np.linalg.cond(basis)), residual]), basis, concentrations, tissue


def fit_controller(images):
    values, failures = [], 0
    for image in images:
        try:
            values.append(diagnostics(image)[0])
        except ValueError:
            failures += 1
    if len(values) < 10:
        raise ValueError("Not enough valid training decompositions")
    low, high = np.quantile(values, [0.1, 0.9], axis=0)
    return {"low": low.tolist(), "high": high.tolist(), "minimum_strength": 0.1,
            "fit_successes": len(values), "fit_failures": failures}


def quality(values, parameters):
    low, high = np.asarray(parameters["low"]), np.asarray(parameters["high"])
    return float(np.clip((values - low) / np.maximum(high - low, 1e-8), 0, 1).mean())


def strength(values, parameters):
    return 1 - (1 - parameters["minimum_strength"]) * quality(values, parameters)


def apply(image, stain_parameters, controller_parameters, rng, fixed_strength=None):
    try:
        values, basis, concentrations, tissue = diagnostics(image)
    except ValueError:
        return image.copy(), {"fallback": True, "strength": 0.0}
    alpha = strength(values, controller_parameters) if fixed_strength is None else fixed_strength
    if not 0 <= alpha <= 1:
        raise ValueError("strength must be between zero and one")
    if alpha == 0:
        return image.copy(), {"fallback": False, "strength": 0.0}
    target = sample(stain_parameters, "spherical", rng)
    scale = 1 + rng.uniform(-0.2, 0.2)
    od = rgb2od(image).reshape(-1, 3)
    # Keep the original decomposition residual; alpha=0 is exactly the input.
    delta = ((target * scale - basis) @ concentrations).T
    updated = od.copy()
    updated[tissue] += alpha * delta[tissue]
    rgb = np.clip(255 * np.exp(-np.maximum(updated, 0)), 0, 255)
    output = np.rint(rgb).reshape(image.shape).astype(np.uint8)
    output.reshape(-1, 3)[~tissue] = image.reshape(-1, 3)[~tissue]
    return output, {"fallback": False, "strength": float(alpha)}
