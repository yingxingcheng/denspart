# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
"""Hirshfeld partitioning with fixed contracted-Gaussian pro-atoms."""

import json
from pathlib import Path

import numpy as np

from .lisa import GaussianFunction, LISAProModel

__all__ = ["GaussianHirshfeldProModel", "load_hirshfeld_basis"]


def load_hirshfeld_basis(source):
    """Load fixed neutral Gaussian pro-atoms from a versioned basis library.

    The state-resolved ``denspart-proatom-basis-v2`` layout is used so the same
    source data can later support charged-state Hirshfeld-I interpolation and AVH.
    Primitive populations are preserved: unlike LISA, they are not initial guesses.
    """
    if source is None:
        raise ValueError("Gaussian Hirshfeld requires a pro-atom basis file.")
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf8") as handle:
            library = json.load(handle)
    else:
        library = source
    if not isinstance(library, dict) or library.get("format") != "denspart-proatom-basis-v2":
        raise ValueError("Expected a denspart-proatom-basis-v2 mapping.")
    elements = library.get("elements")
    if not isinstance(elements, dict) or not elements:
        raise ValueError("The pro-atom basis contains no elements.")

    result = {}
    for raw_atnum, element in elements.items():
        try:
            atnum = int(raw_atnum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in pro-atom basis: {raw_atnum!r}.") from exc
        neutral_states = [state for state in element.get("states", []) if state.get("charge") == 0]
        if len(neutral_states) != 1:
            raise ValueError(f"Atomic number {atnum} must have exactly one neutral state.")
        state = neutral_states[0]
        primitives = state.get("primitives", {})
        orders = np.asarray(primitives.get("orders"), dtype=float)
        exponents = np.asarray(primitives.get("exponents"), dtype=float)
        populations = np.asarray(primitives.get("populations"), dtype=float)
        if orders.ndim != 1 or exponents.ndim != 1 or populations.ndim != 1:
            raise ValueError(f"Neutral pro-atom arrays for atomic number {atnum} must be 1D.")
        if not (len(orders) == len(exponents) == len(populations)) or len(orders) == 0:
            raise ValueError(
                f"Neutral pro-atom arrays for atomic number {atnum} have unequal lengths."
            )
        if not np.all(orders == 2):
            raise ValueError("DensPart Gaussian Hirshfeld supports only order-two functions.")
        if not np.isfinite(exponents).all() or not (exponents > 0).all():
            raise ValueError(f"Exponents for atomic number {atnum} must be finite and positive.")
        if not np.isfinite(populations).all() or not (populations >= 0).all():
            raise ValueError(
                f"Populations for atomic number {atnum} must be finite and nonnegative."
            )
        if not np.isclose(populations.sum(), atnum, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                f"Neutral populations for atomic number {atnum} integrate to "
                f"{populations.sum():.12g}, expected {atnum}."
            )
        result[atnum] = (exponents, populations)
    return result


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
