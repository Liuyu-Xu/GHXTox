import numpy as np

from ghxtox.bioemu_conformer_cache import bioemu_ensemble_descriptors
from ghxtox.conformer_cache import ensemble_descriptors


def test_bioemu_descriptors_convert_nm_to_angstrom() -> None:
    coords_nm = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.4, 0.0]],
        ],
        dtype=np.float32,
    )
    expected = ensemble_descriptors(coords_nm * 10.0)
    actual = bioemu_ensemble_descriptors(coords_nm)
    assert set(actual) == set(expected)
    for key in expected:
        assert np.allclose(actual[key], expected[key])
    assert actual["contact_occupancy_6"][0, 1] == 1.0
