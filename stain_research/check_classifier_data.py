"""Behavioral checks for test exclusion and deterministic validation loading."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from classifier_pilot import Images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    prepared = args.run / 'prepared'
    parameters = json.loads((prepared / 'parameters.json').read_text())
    val = pd.read_csv(prepared / 'val.csv').head(1)
    dataset = Images(val, parameters, 'adaptive', False)
    torch.manual_seed(123)
    first = dataset[0][0]
    torch.manual_seed(456)
    second = dataset[0][0]
    assert torch.equal(first, second), 'Validation depends on random state'
    assert first.shape == (16, 3, 56, 56)
    assert torch.isfinite(first).all()
    forbidden = pd.DataFrame([{'split': 'test', 'image_path': 'MUST_NOT_BE_OPENED', 'class_code': 'TUM'}])
    try:
        Images(forbidden, parameters, 'adaptive', False)
    except ValueError as exc:
        assert 'Test images are forbidden' in str(exc)
    else:
        raise AssertionError('Test loader was allowed')
    result = {'validation_rng_independent': True, 'patch_sequence_shape': list(first.shape),
              'test_loader_rejected_before_file_access': True}
    (args.run / 'loader_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
