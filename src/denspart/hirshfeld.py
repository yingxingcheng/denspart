# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
"""Hirshfeld partitioning with fixed contracted-Gaussian pro-atoms."""

import json
from pathlib import Path

import numpy as np

from .lisa import GaussianFunction, LISAProModel

__all__ = ["GaussianHirshfeldProModel", "load_hirshfeld_basis", "load_proatom_states"]


def _load_library(source, expected_format):
    """Return a JSON-like basis mapping and validate its format marker."""
    if source is None:
        raise ValueError(f"A {expected_format} basis file is required.")
    if isinstance(source, str | Path):
        with Path(source).open(encoding="utf8") as handle:
            library = json.load(handle)
    else:
        library = source
    if not isinstance(library, dict) or library.get("format") != expected_format:
        raise ValueError(f"Expected a {expected_format} mapping.")
    elements = library.get("elements")
    if not isinstance(elements, dict) or not elements:
        raise ValueError("The pro-atom basis contains no elements.")
    return library, elements


def _validate_state_primitives(atnum, charge, state, key="primitives", normalized=False):
    """Validate and return one Gaussian atomic-state expansion."""
    primitives = state.get(key, {})
    orders = np.asarray(primitives.get("orders"), dtype=float)
    exponents = np.asarray(primitives.get("exponents"), dtype=float)
    value_key = "coefficients" if normalized else "populations"
    values = np.asarray(primitives.get(value_key), dtype=float)
    if orders.ndim != 1 or exponents.ndim != 1 or values.ndim != 1:
        raise ValueError(f"Pro-atom arrays for Z={atnum}, charge={charge:+d} must be 1D.")
    if not (len(orders) == len(exponents) == len(values)):
        raise ValueError(f"Pro-atom arrays for Z={atnum}, charge={charge:+d} have unequal lengths.")
    expected_population = 1 if normalized else atnum - charge
    if expected_population < 0:
        raise ValueError(f"Charge {charge:+d} gives a negative population for Z={atnum}.")
    if expected_population > 0 and len(orders) == 0:
        raise ValueError(f"Populated state Z={atnum}, charge={charge:+d} has no primitives.")
    if not np.all(orders == 2):
        raise ValueError("DensPart Gaussian pro-atoms support only order-two functions.")
    if not np.isfinite(exponents).all() or not (exponents > 0).all():
        raise ValueError(
            f"Exponents for Z={atnum}, charge={charge:+d} must be finite and positive."
        )
    if not np.isfinite(values).all() or not (values >= 0).all():
        raise ValueError(
            f"{value_key.capitalize()} for Z={atnum}, charge={charge:+d} must be "
            "finite and nonnegative."
        )
    expected_sum = float(expected_population)
    if not np.isclose(values.sum(), expected_sum, rtol=0.0, atol=1.0e-8):
        raise ValueError(
            f"State Z={atnum}, charge={charge:+d} {value_key} integrate to "
            f"{values.sum():.12g}, expected {expected_sum:.12g}."
        )
    return exponents, values


def load_proatom_states(source):
    """Load all Gaussian charge states from a state-preserving basis library."""
    _, elements = _load_library(source, "denspart-proatom-basis-v2")
    result = {}
    for raw_atnum, element in elements.items():
        try:
            atnum = int(raw_atnum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in pro-atom basis: {raw_atnum!r}.") from exc
        states = {}
        for state in element.get("states", []):
            raw_charge = state.get("charge")
            if not isinstance(raw_charge, int | np.integer):
                raise ValueError(f"Charge states for atomic number {atnum} must be integers.")
            charge = int(raw_charge)
            if charge in states:
                raise ValueError(f"Duplicate charge {charge:+d} for atomic number {atnum}.")
            electrons = state.get("electrons", atnum - charge)
            if int(electrons) != atnum - charge:
                raise ValueError(
                    f"State Z={atnum}, charge={charge:+d} has inconsistent electron count."
                )
            states[charge] = _validate_state_primitives(atnum, charge, state)
        if 0 not in states:
            raise ValueError(f"Atomic number {atnum} must have exactly one neutral state.")
        result[atnum] = states
    return result


def load_hirshfeld_basis(source):
    """Load fixed neutral Gaussian pro-atoms from a versioned basis library.

    The state-resolved ``denspart-proatom-basis-v2`` layout is used so the same
    source data can later support charged-state Hirshfeld-I interpolation and AVH.
    Primitive populations are preserved: unlike LISA, they are not initial guesses.
    """
    return {atnum: states[0] for atnum, states in load_proatom_states(source).items()}


class GaussianHirshfeldProModel(LISAProModel):
    """Fixed neutral pro-density model for conventional Hirshfeld partitioning."""

    @classmethod
    def from_geometry(cls, atnums, atcoords, basis=None):
        """Construct fixed neutral Gaussian pro-atoms for a geometry."""
        basis = load_hirshfeld_basis(basis)
        functions = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            atnum = int(atnum)
            if atnum not in basis:
                available = ", ".join(str(number) for number in sorted(basis))
                raise NotImplementedError(
                    f"No neutral Gaussian pro-atom is available for atomic number {atnum}. "
                    f"Available atomic numbers: {available}."
                )
            exponents, populations = basis[atnum]
            for population, exponent in zip(populations, exponents, strict=True):
                functions.append(GaussianFunction(iatom, atcoord, [population], exponent))
        return cls(np.asarray(atnums), np.asarray(atcoords), functions)

    def to_dict(self):
        """Return a reconstructible fixed-reference model."""
        result = super().to_dict()
        result["method"] = np.array("HIRSHFELD")
        result["reference_charges"] = self.charges
        return result

    @classmethod
    def from_dict(cls, data):
        """Reconstruct a fixed Gaussian Hirshfeld model from stored arrays."""
        if str(data["class"]) != "GaussianHirshfeldProModel":
            raise TypeError("Expected a GaussianHirshfeldProModel dictionary.")
        functions = []
        ipar = 0
        atnums = np.asarray(data["atnums"])
        atcoords = np.asarray(data["atcoords"])
        atnfns = np.asarray(data["atnfns"], dtype=int)
        populations = np.asarray(data["propars"], dtype=float)
        exponents = np.asarray(data["exponents"], dtype=float)
        for iatom, atcoord in enumerate(atcoords):
            for _ in range(atnfns[iatom]):
                functions.append(
                    GaussianFunction(iatom, atcoord, [populations[ipar]], exponents[ipar])
                )
                ipar += 1
        if ipar != len(populations) or ipar != len(exponents):
            raise ValueError(
                "Stored Gaussian Hirshfeld primitive arrays have inconsistent lengths."
            )
        return cls(atnums, atcoords, functions)
