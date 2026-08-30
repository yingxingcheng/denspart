# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
#
# DensPart is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
"""Unit tests for the LISA pro-density model."""

import json

import numpy as np
import pytest
from grid.periodicgrid import PeriodicGrid

from denspart import ProModel
from denspart.cache import ComputeCache
from denspart.lisa import GaussianFunction, LISAProModel, load_lisa_basis


def test_gaussian_validation_and_cutoff():
    with pytest.raises(ValueError, match="nonnegative"):
        GaussianFunction(0, np.zeros(3), [-1.0], 1.0)
    with pytest.raises(ValueError, match="positive Gaussian exponent"):
        GaussianFunction(0, np.zeros(3), [1.0], 0.0)

    function = GaussianFunction(0, np.zeros(3), [0.0], 1.0)
    assert function.get_cutoff_radius(1e-10) == 0.0
    assert np.isinf(function.get_cutoff_radius(0.0))


def test_cache_distinguishes_point_arrays():
    function = GaussianFunction(0, np.zeros(3), [1.0], 1.0)
    points1 = np.zeros((2, 3))
    points2 = np.ones((2, 3))
    cache = ComputeCache()

    assert not np.allclose(function.compute(points1, cache), function.compute(points2, cache))
    np.testing.assert_allclose(
        function.compute(points2, cache),
        function.compute(points2, None),
    )
    assert cache._objects[id(points1)] is points1
    assert cache._objects[id(points2)] is points2


def test_cache_distinguishes_centers_on_shared_points():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    function1 = GaussianFunction(0, np.zeros(3), [1.0], 1.0)
    function2 = GaussianFunction(1, np.ones(3), [1.0], 1.0)
    cache = ComputeCache()
    np.testing.assert_allclose(function1.compute(points, cache), function1.compute(points, None))
    np.testing.assert_allclose(function2.compute(points, cache), function2.compute(points, None))


def test_periodic_local_grid_uses_nearest_image():
    grid = PeriodicGrid(
        np.array([[0.1, 0.2, 0.2], [2.0, 2.0, 2.0]]),
        np.ones(2),
        np.eye(3) * 4.0,
        wrap=True,
    )
    center = np.array([3.9, 0.2, 0.2])
    localgrid = grid.get_localgrid(center, 0.5)
    function = GaussianFunction(0, center, [1.0], 1.0)

    np.testing.assert_equal(localgrid.indices, [0])
    np.testing.assert_allclose(localgrid.points, [[4.1, 0.2, 0.2]])
    expected = (1 / np.pi) ** 1.5 * np.exp(-(0.2**2))
    np.testing.assert_allclose(function.compute(localgrid.points), [expected])


def test_derivative_finite_difference():
    function = GaussianFunction(0, np.zeros(3), [1.3], 0.7)
    points = np.array([[0.0, 0.0, 0.0], [0.2, -0.4, 0.1]])
    step = 1e-6
    analytic = function.compute_derivatives(points)[0]
    function.pars[0] += step
    plus = function.compute(points)
    function.pars[0] -= 2 * step
    minus = function.compute(points)
    function.pars[0] += step
    np.testing.assert_allclose(analytic, (plus - minus) / (2 * step), rtol=1e-9, atol=1e-11)


def test_load_legacy_and_versioned_basis(tmp_path):
    legacy = {"1": [[2, 2], [2.0, 0.5], [3.0, 1.0]]}
    filename = tmp_path / "basis.json"
    filename.write_text(json.dumps(legacy), encoding="utf8")
    exponents, initials = load_lisa_basis(filename)[1]
    np.testing.assert_equal(exponents, [2.0, 0.5])
    np.testing.assert_allclose(initials, [0.75, 0.25])

    versioned = {
        "format": "aim-lisa-basis-v1",
        "metadata": {"method": "test"},
        "elements": {"1": {"orders": [2], "exponents": [1.0], "initials": [2.0]}},
    }
    exponents, initials = load_lisa_basis(versioned)[1]
    np.testing.assert_equal(exponents, [1.0])
    np.testing.assert_equal(initials, [1.0])
    versioned["format"] = "denspart-lisa-basis-v1"
    assert 1 in load_lisa_basis(versioned)


@pytest.mark.parametrize(
    "basis, message",
    [
        ({"1": [[1], [1.0], [1.0]]}, "order-two"),
        ({"1": [[2], [-1.0], [1.0]]}, "finite and positive"),
        ({"1": [[2], [1.0], [0.0]]}, "not all zero"),
    ],
)
def test_reject_invalid_basis(basis, message):
    with pytest.raises(ValueError, match=message):
        load_lisa_basis(basis)


def test_initial_population_and_missing_element():
    model = LISAProModel.from_geometry(np.array([1, 3, 5]), np.zeros((3, 3)))
    np.testing.assert_allclose(
        [
            sum(function.population for function in model.fns if function.iatom == iatom)
            for iatom in range(3)
        ],
        [1.0, 3.0, 5.0],
    )
    with pytest.raises(NotImplementedError, match="custom basis file"):
        LISAProModel.from_geometry(np.array([2]), np.zeros((1, 3)))


def test_gap_default_basis():
    model = LISAProModel.from_geometry(np.array([31, 15]), np.zeros((2, 3)))
    assert [sum(function.iatom == iatom for function in model.fns) for iatom in range(2)] == [
        14,
        15,
    ]
    np.testing.assert_allclose(
        [
            sum(function.population for function in model.fns if function.iatom == iatom)
            for iatom in range(2)
        ],
        [31.0, 15.0],
    )


def test_reduction_is_deterministic_and_roundtrips():
    functions = [
        GaussianFunction(0, np.zeros(3), [0.4], 2.0),
        GaussianFunction(0, np.zeros(3), [0.6], 2.0001),
        GaussianFunction(0, np.zeros(3), [1.0], 0.5),
    ]
    model = LISAProModel(np.array([1]), np.zeros((1, 3)), functions).reduce(eps=1e-3)
    assert [function.exponent for function in model.fns] == pytest.approx([2.00005, 0.5])
    assert [function.population for function in model.fns] == pytest.approx([1.0, 1.0])

    restored = LISAProModel.from_dict(model.to_dict())
    assert [function.exponent for function in restored.fns] == pytest.approx([2.00005, 0.5])
    assert [function.population for function in restored.fns] == pytest.approx([1.0, 1.0])


def test_base_class_reconstruction(tmp_path):
    model = LISAProModel.from_geometry(np.array([1]), np.zeros((1, 3)))
    filename = tmp_path / "lisa.npz"
    np.savez(filename, **model.to_dict())
    restored = ProModel.from_dict(np.load(filename))
    assert isinstance(restored, LISAProModel)
    np.testing.assert_allclose(
        [function.exponent for function in restored.fns],
        [function.exponent for function in model.fns],
    )
