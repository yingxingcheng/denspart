"""Tests for Gaussian charged-state iterative Hirshfeld partitioning."""

import json

import numpy as np
import pytest
from grid.basegrid import Grid

from denspart import ProModel
from denspart.__main__ import main
from denspart.hirshfeld import load_proatom_states
from denspart.hirshfeld_i import GaussianHirshfeldIProModel, optimize_hirshfeld_i


def make_state_basis():
    """Return three simple Gaussian charge states for a helium-like atom."""
    states = []
    for charge, exponent in ((1, 2.0), (0, 0.8), (-1, 0.3)):
        electrons = 2 - charge
        states.append(
            {
                "charge": charge,
                "electrons": electrons,
                "primitives": {
                    "orders": [2],
                    "exponents": [exponent],
                    "populations": [electrons],
                },
            }
        )
    return {
        "format": "denspart-proatom-basis-v2",
        "metadata": {},
        "elements": {"2": {"symbol": "He", "states": states}},
    }


def make_radial_grid(npoint=1200):
    """Return a one-ray spherical quadrature for radial test densities."""
    radii = np.linspace(1.0e-6, 12.0, npoint)
    dr = radii[1] - radii[0]
    weights = 4 * np.pi * radii**2 * dr
    weights[[0, -1]] *= 0.5
    points = np.column_stack([radii, np.zeros((npoint, 2))])
    return Grid(points, weights)


def test_load_states_and_interpolate():
    states = load_proatom_states(make_state_basis())[2]
    assert sorted(states) == [-1, 0, 1]
    model = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    function = model.fns[0]
    function.pars[0] = 0.25
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected = 0.75 * function._compute_state(0, points)
    expected += 0.25 * function._compute_state(1, points)
    np.testing.assert_allclose(function.compute(points), expected)
    assert function.population == pytest.approx(1.75)
    assert model.charges == pytest.approx([0.25])


def test_cutoff_uses_only_active_charge_states():
    model = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    function = model.fns[0]
    neutral_radius = function.get_cutoff_radius(1.0e-10)
    function.pars[0] = -0.2
    anionic_radius = function.get_cutoff_radius(1.0e-10)
    assert anionic_radius > neutral_radius


def test_hirshfeld_i_rejects_invalid_mixing():
    grid = make_radial_grid(20)
    model = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    with pytest.raises(ValueError, match="mixing"):
        optimize_hirshfeld_i(model, grid, np.ones(20), mixing=0.0)


def test_interpolated_model_roundtrip():
    model = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    model.fns[0].pars[0] = -0.4
    restored = ProModel.from_dict(model.to_dict())
    assert isinstance(restored, GaussianHirshfeldIProModel)
    np.testing.assert_allclose(restored.charges, [-0.4])
    points = np.array([[0.2, 0.0, 0.0], [0.7, 0.0, 0.0]])
    np.testing.assert_allclose(
        restored.compute_proatom(0, points), model.compute_proatom(0, points)
    )


def test_hirshfeld_i_converges_to_atomic_population(tmp_path):
    grid = make_radial_grid()
    target = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    target.fns[0].pars[0] = 0.35
    density = target.compute_density(grid)
    expected_charge = 2.0 - grid.integrate(density)

    model = GaussianHirshfeldIProModel.from_geometry(
        np.array([2]), np.zeros((1, 3)), make_state_basis()
    )
    model, _ = optimize_hirshfeld_i(
        model, grid, density, threshold=1.0e-10, maxiter=10, density_cutoff=0.0
    )
    assert model.charges == pytest.approx([expected_charge], abs=1.0e-10)
    assert len(model.charge_history) == 3

    basis_file = tmp_path / "states.json"
    basis_file.write_text(json.dumps(make_state_basis()), encoding="utf8")
    density_file = tmp_path / "density.npz"
    output_file = tmp_path / "hirshfeld-i.npz"
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
            "HIRSHFELD-I",
            "--proatom-basis",
            str(basis_file),
            "--density-cutoff",
            "0",
        ]
    )
    with np.load(output_file) as result:
        assert result["method"] == "HIRSHFELD-I"
        np.testing.assert_allclose(result["charges"], [expected_charge], atol=1.0e-8)


def test_hirshfeld_i_rejects_missing_and_out_of_range_states():
    basis = make_state_basis()
    model = GaussianHirshfeldIProModel.from_geometry(np.array([2]), np.zeros((1, 3)), basis)
    del model.fns[0].states[1]
    model.fns[0].pars[0] = 0.2
    with pytest.raises(ValueError, match="Missing Hirshfeld-I state"):
        model.compute_proatom(0, np.zeros((1, 3)))
    model.fns[0].pars[0] = -1.2
    with pytest.raises(ValueError, match="outside the available range"):
        model.compute_proatom(0, np.zeros((1, 3)))
