"""Numerical compatibility with selected upstream SVD functions, plus edge cases."""
import ast
from pathlib import Path
import numpy as np
from PIL import Image
from transforms import extract, fit, reconstruct, rgb2od, transform
from controller import apply, strength


def upstream_functions():
    # Load only reviewed mathematical functions; avoid optional spams/TensorFlow imports.
    namespace = {"np": np}
    for filename, names in [("utils.py", {"get_mask", "reconstruct_image", "od2rgb"}),
                            ("StainSAN.py", {"get_stain_svd"})]:
        tree = ast.parse((Path(__file__).parent / "upstream/stain_san" / filename).read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in selected:
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr == "Inf":
                    child.attr = "inf"
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    namespace["rgb2od"] = lambda image, Io=255: rgb2od(image)
    return namespace


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    upstream = upstream_functions()
    matrices = []
    for code in ("ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"):
        path = args.project / f"data/raw/HMU-GC-HE-30K/all_image/{code}/{code}_1.png"
        with Image.open(path) as source:
            image = np.array(source.convert("RGB"))
        basis = extract(image)
        reference, _, _ = upstream["get_stain_svd"](image)
        assert np.allclose(basis, reference, atol=1e-10), code
        matrices.append(basis)
        od = rgb2od(image).reshape(-1, 3)
        h = np.linalg.lstsq(basis, od.T, rcond=None)[0]
        expected = upstream["reconstruct_image"](basis, h, od, 0.3, np.inf, image.shape)
        assert np.array_equal(reconstruct(image, basis, basis), expected), code
    parameters = fit(matrices * 2)
    for mode in ("spherical", "elliptical", "uniform"):
        blank = np.full((32, 32, 3), 255, dtype=np.uint8)
        unchanged, failed = transform(blank, parameters, mode, np.random.default_rng(1))
        assert failed and np.array_equal(blank, unchanged)
        one, _ = transform(image, parameters, mode, np.random.default_rng(1), training=False)
        two, _ = transform(image, parameters, mode, np.random.default_rng(2), training=False)
        assert np.array_equal(one, two)
    assert np.isclose(np.trace(parameters["covariance"]), 6 * parameters["variance"])
    controller = {"low": [1.0, 0.02], "high": [2.0, 0.1], "minimum_strength": 0.1}
    unchanged, _ = apply(image, parameters, controller, np.random.default_rng(1), 0)
    assert np.array_equal(image, unchanged)
    one, _ = apply(image, parameters, controller, np.random.default_rng(1))
    two, _ = apply(image, parameters, controller, np.random.default_rng(1))
    assert np.array_equal(one, two)
    assert strength(np.array([1.0, 0.02]), controller) == 1.0
    assert np.isclose(strength(np.array([2.0, 0.1]), controller), 0.1)
    unchanged, meta = apply(blank, parameters, controller, np.random.default_rng(1))
    assert meta["fallback"] and np.array_equal(unchanged, blank)
    print("PASS: controller identity, repeatability, monotonic limits, blank fallback")
    print("PASS: upstream SVD/reconstruction match for 8 classes, blank fallback, deterministic evaluation, covariance trace")


if __name__ == "__main__":
    main()
