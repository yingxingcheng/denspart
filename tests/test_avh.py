"""Tests for Additive Variational Hirshfeld partitioning."""

import json

import numpy as np
import pytest
from grid.basegrid import Grid

from denspart import ProModel
from denspart.__main__ import main
from denspart.avh import AVHProModel, load_avh_basis, optimize_avh_pro_model
from denspart.cache import ComputeCache


def make_avh_basis():
    """Return neutral and cationic normalized Gaussian shapes."""
    return {
        "format": "denspart-avh-basis-v1",
        "metadata": {"variant": "test"},
        "elements": {
            "2": {
                "symbol": "He",
                "states": [
                    {
                        "charge": 1,
                        "electrons": 1,
                        "shape_primitives": {
                            "orders": [2],
                            "exponents": [2.0],
                            "coefficients": [1.0],
                        },
                    },
                    {
                        "charge": 0,
                        "electrons": 2,
                        "shape_primitives": {
                            "orders": [2],
                            "exponents": [0.5],
                            "coefficients": [1.0],
                        },
                    },
                ],
            }
        },
    }


def make_radial_grid(npoint=1600):
    """Return a one-ray spherical quadrature for radial test densities."""
    radii = np.linspace(1.0e-6, 14.0, npoint)
    dr = radii[1] - radii[0]
    weights = 4 * np.pi * radii**2 * dr
    weights[[0, -1]] *= 0.5
    points = np.column_stack([radii, np.zeros((npoint, 2))])
    return Grid(points, weights)


def test_load_avh_basis_and_initial_model():
    states = load_avh_basis(make_avh_basis())[2]
    assert [state[0] for state in states] == [1, 0]
    model = AVHProModel.from_geometry(np.array([2]), np.zeros((1, 3)), make_avh_basis())
    assert [function.population for function in model.fns] == pytest.approx([0.0, 2.0])
    assert [function.state_multiplier for function in model.fns] == pytest.approx([0.0, 1.0])
    assert model.charges == pytest.approx([0.0])
    model.assign_pars(np.zeros(2))
    np.testing.assert_allclose(model.get_cutoff_radii(1.0e-10), [0.0])


def test_avh_optimizer_and_roundtrip(tmp_path):
    grid = make_radial_grid()
    target = AVHProModel.from_geometry(np.array([2]), np.zeros((1, 3)), make_avh_basis())
    target.assign_pars(np.array([0.6, 1.1]))
    density = target.compute_density(grid)

    model = AVHProModel.from_geometry(np.array([2]), np.zeros((1, 3)), make_avh_basis())
    model, _ = optimize_avh_pro_model(
        model, grid, density, gtol=1.0e-12, maxiter=100, density_cutoff=0.0
    )
    np.testing.assert_allclose(
        [function.population for function in model.fns], [0.6, 1.1], atol=1e-6
    )
    assert model.population == pytest.approx(grid.integrate(density), abs=1.0e-12)
    np.testing.assert_allclose(model.charges, [0.3], atol=1e-6)
    restored = ProModel.from_dict(model.to_dict())
    assert isinstance(restored, AVHProModel)
    np.testing.assert_allclose(restored.compute_density(grid), model.compute_density(grid))

    basis_file = tmp_path / "avh.json"
    basis_file.write_text(json.dumps(make_avh_basis()), encoding="utf8")
    density_file = tmp_path / "density.npz"
    output_file = tmp_path / "avh.npz"
    np.savez(
        density_file,
        atnums=np.array([2]),
        atcoords=np.zeros((1, 3)),
        density=density,
        points=grid.points,
        weights=grid.weights,
        cellvecs=np.zeros((0, 3)),
    )
    main(
        [
            str(density_file),
            str(output_file),
            "--method",
            "AVH",
            "--avh-basis",
            str(basis_file),
            "--density-cutoff",
            "0",
            "--gtol",
            "1e-12",
        ]
    )
    with np.load(output_file) as result:
        assert result["method"] == "AVH"
        np.testing.assert_allclose(result["charges"], [0.3], atol=1e-6)
        assert (result["propars"] >= 0).all()


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("coefficients", [0.9], "integrate"),
        ("orders", [1], "order-two"),
        ("exponents", [-1.0], "finite and positive"),
    ],
)
def test_reject_invalid_avh_basis(field, value, message):
    basis = make_avh_basis()
    basis["elements"]["2"]["states"][0]["shape_primitives"][field] = value
    with pytest.raises(ValueError, match=message):
        load_avh_basis(basis)


def test_cache_discard_missing_stage_is_safe():
    cache = ComputeCache()
    cache.keep("forever", ("value",), np.ones(2))
    cache.discard("missing")
    cache.clear()
    assert cache.fetch("forever", ("value",)) is None
