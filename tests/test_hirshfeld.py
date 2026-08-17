"""Tests for fixed Gaussian-reference Hirshfeld partitioning."""

import json

import numpy as np
import pytest

from denspart import ProModel
from denspart.__main__ import main
from denspart.hirshfeld import GaussianHirshfeldProModel, load_hirshfeld_basis
from denspart.lisa import DEFAULT_LISA_BASIS


def make_basis():
    """Return a minimal state-preserving hydrogen library."""
    return {
        "format": "denspart-proatom-basis-v2",
        "metadata": {},
        "elements": {
            "1": {
                "symbol": "H",
                "states": [
                    {
                        "charge": 0,
                        "electrons": 1,
                        "multiplicity": 2,
                        "primitives": {
                            "orders": [2, 2],
                            "exponents": [2.0, 0.5],
                            "populations": [0.25, 0.75],
                        },
                    },
                    {
                        "charge": 1,
                        "electrons": 0,
                        "multiplicity": 1,
                        "primitives": {"orders": [], "exponents": [], "populations": []},
                    },
                ],
            }
        },
    }


def make_water_basis():
    """Build neutral H/O references from the established Gaussian basis."""
    basis = make_basis()
    for atnum, symbol, multiplicity in ((1, "H", 2), (8, "O", 3)):
        orders, exponents, populations = DEFAULT_LISA_BASIS[atnum]
        populations = np.asarray(populations) * atnum / np.sum(populations)
        basis["elements"][str(atnum)] = {
            "symbol": symbol,
            "states": [
                {
                    "charge": 0,
                    "electrons": atnum,
                    "multiplicity": multiplicity,
                    "primitives": {
                        "orders": orders,
                        "exponents": exponents,
                        "populations": populations.tolist(),
                    },
                }
            ],
        }
    return basis


def test_load_fixed_neutral_state(tmp_path):
    filename = tmp_path / "proatoms.json"
    filename.write_text(json.dumps(make_basis()), encoding="utf8")
    exponents, populations = load_hirshfeld_basis(filename)[1]
    np.testing.assert_equal(exponents, [2.0, 0.5])
    np.testing.assert_equal(populations, [0.25, 0.75])


@pytest.mark.parametrize(
    "change, message",
    [
        (("format", "wrong"), "denspart-proatom-basis-v2"),
        (("population", [0.2, 0.7]), "integrate"),
        (("orders", [1, 2]), "order-two"),
    ],
)
def test_reject_invalid_basis(change, message):
    basis = make_basis()
    key, value = change
    if key == "format":
        basis["format"] = value
    else:
        field = "populations" if key == "population" else key
        basis["elements"]["1"]["states"][0]["primitives"][field] = value
    with pytest.raises(ValueError, match=message):
        load_hirshfeld_basis(basis)


def test_fixed_model_density_and_roundtrip():
    model = GaussianHirshfeldProModel.from_geometry(np.array([1]), np.zeros((1, 3)), make_basis())
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    before = model.compute_proatom(0, points)
    restored = ProModel.from_dict(model.to_dict())
    assert isinstance(restored, GaussianHirshfeldProModel)
    np.testing.assert_allclose(restored.compute_proatom(0, points), before)
    np.testing.assert_allclose(restored.charges, [0.0])


def test_cli_reports_partitioned_not_reference_charges(tmp_path):
    basis_file = tmp_path / "proatoms.json"
    basis_file.write_text(json.dumps(make_water_basis()), encoding="utf8")
    output = tmp_path / "hirshfeld.npz"
    main(
        [
            "tests/density-water.npz",
            str(output),
            "--method",
            "HIRSHFELD",
            "--proatom-basis",
            str(basis_file),
            "--density-cutoff",
            "0",
        ]
    )
    with np.load("tests/density-water.npz") as density, np.load(output) as result:
        expected_total_charge = density["atnums"].sum() - np.dot(
            density["density"], density["weights"]
        )
        assert result["method"] == "HIRSHFELD"
        np.testing.assert_allclose(result["reference_charges"], 0.0, atol=1.0e-12)
        assert result["charges"].sum() == pytest.approx(expected_total_charge, abs=5.0e-7)
