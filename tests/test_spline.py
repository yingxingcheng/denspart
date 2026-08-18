"""Tests for shared radial-spline stockholder models."""

import numpy as np
import pytest
from grid.basegrid import Grid

from denspart import ProModel
from denspart.avh import optimize_avh_pro_model
from denspart.spline import (
    SplineProModel,
    load_spline_basis,
    optimize_spline_hirshfeld_i,
)


def make_spline_basis():
    """Return normalized H-, H, and H+ radial densities."""
    radii = np.linspace(1.0e-4, 12.0, 801)
    step = radii[1] - radii[0]
    radial_weights = np.full_like(radii, step)
    radial_weights[[0, -1]] *= 0.5
    volume_weights = 4.0 * np.pi * radii**2 * radial_weights

    states = []
    for charge, exponent in ((-1, 0.4), (0, 1.0), (1, None)):
        electrons = 1 - charge
        if electrons:
            density = np.exp(-exponent * radii**2)
            density *= electrons / np.dot(volume_weights, density)
        else:
            density = np.zeros_like(radii)
        states.append({"charge": charge, "electrons": electrons, "density": density.tolist()})
    return {
        "format": "denspart-spline-proatom-basis-v1",
        "metadata": {},
        "elements": {
            "1": {
                "symbol": "H",
                "radii": radii.tolist(),
                "radial_weights": radial_weights.tolist(),
                "states": states,
            }
        },
    }


def make_radial_grid(basis):
    """Represent a spherical radial quadrature as a DensPart integration grid."""
    element = basis["elements"]["1"]
    radii = np.asarray(element["radii"])
    radial_weights = np.asarray(element["radial_weights"])
    points = np.zeros((len(radii), 3))
    points[:, 0] = radii
    weights = 4.0 * np.pi * radii**2 * radial_weights
    return Grid(points, weights)


def test_load_spline_basis_and_reject_bad_normalization():
    basis = make_spline_basis()
    states = load_spline_basis(basis)[1]
    assert [state[0] for state in states] == [-1, 0, 1]
    basis["elements"]["1"]["states"][1]["density"][10] *= 2.0
    with pytest.raises(ValueError, match="integrates to"):
        load_spline_basis(basis)


def test_shared_coefficients_and_roundtrip():
    basis = make_spline_basis()
    model = SplineProModel.from_geometry(
        np.array([1]), np.zeros((1, 3)), basis, method="HIRSHFELD-I"
    )
    model.set_hirshfeld_i_charges(np.array([-0.25]))
    functions = {function.charge: function for function in model.fns}
    assert functions[-1].population == pytest.approx(0.5)
    assert functions[0].population == pytest.approx(0.75)
    assert functions[1].population == pytest.approx(0.0)
    np.testing.assert_allclose(model.charges, [-0.25], atol=1.0e-12)

    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [15.0, 0.0, 0.0]])
    density = model.compute_proatom(0, points)
    assert np.all(density >= 0.0)
    assert density[-1] == 0.0
    restored = ProModel.from_dict(model.to_dict())
    assert isinstance(restored, SplineProModel)
    assert restored.method == "HIRSHFELD-I"
    np.testing.assert_allclose(restored.compute_proatom(0, points), density)
    np.testing.assert_allclose(restored.charges, model.charges)


def test_spline_hirshfeld_i_recovers_one_atom_population():
    basis = make_spline_basis()
    grid = make_radial_grid(basis)
    target = SplineProModel.from_geometry(
        np.array([1]), np.zeros((1, 3)), basis, method="HIRSHFELD-I"
    )
    target.set_hirshfeld_i_charges(np.array([-0.3]))
    density = target.compute_density(grid)
    model = SplineProModel.from_geometry(
        np.array([1]), np.zeros((1, 3)), basis, method="HIRSHFELD-I"
    )
    model, _ = optimize_spline_hirshfeld_i(
        model, grid, density, threshold=1.0e-10, maxiter=10, density_cutoff=1.0e-8
    )
    # This also exercises rebuilding the local grid when the active pair changes
    # from neutral-only to the anion/neutral pair.
    np.testing.assert_allclose(model.charges, [-0.3], atol=2.0e-6)


def test_hirshfeld_i_uses_implicit_stripped_atom_endpoint():
    basis = make_spline_basis()
    basis["elements"]["1"]["states"] = [
        state for state in basis["elements"]["1"]["states"] if state["charge"] != 1
    ]
    model = SplineProModel.from_geometry(
        np.array([1]), np.zeros((1, 3)), basis, method="HIRSHFELD-I"
    )
    model.set_hirshfeld_i_charges(np.array([0.25]))
    np.testing.assert_allclose(model.charges, [0.25])
    assert sum(function.population for function in model.fns) == pytest.approx(0.75)


def test_spline_avh_optimizes_the_same_state_coefficients():
    basis = make_spline_basis()
    grid = make_radial_grid(basis)
    target = SplineProModel.from_geometry(np.array([1]), np.zeros((1, 3)), basis, method="AVH")
    target.assign_pars(np.array([0.6, 0.7]))
    density = target.compute_density(grid)
    model = SplineProModel.from_geometry(np.array([1]), np.zeros((1, 3)), basis, method="AVH")
    model, _ = optimize_avh_pro_model(
        model, grid, density, gtol=1.0e-10, maxiter=100, density_cutoff=0.0
    )
    np.testing.assert_allclose(model.population, 1.3, atol=1.0e-8)
    np.testing.assert_allclose(
        [function.population for function in model.fns[:2]],
        [0.6, 0.7],
        atol=2.0e-5,
    )
