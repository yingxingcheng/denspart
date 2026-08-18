# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
"""Iterative Hirshfeld partitioning with Gaussian atomic charge states."""

import numpy as np

from .cache import compute_cached
from .hirshfeld import load_proatom_states
from .vh import BasisFunction, ProModel

__all__ = [
    "GaussianHirshfeldIProModel",
    "InterpolatedGaussianFunction",
    "optimize_hirshfeld_i",
]


class InterpolatedGaussianFunction(BasisFunction):
    """One pro-atom obtained by interpolating adjacent integer charge states."""

    def __init__(self, iatom, atnum, center, charge, states):
        self.atnum = int(atnum)
        self.states = {
            int(state_charge): (
                np.asarray(exponents, dtype=float),
                np.asarray(populations, dtype=float),
            )
            for state_charge, (exponents, populations) in states.items()
        }
        if 0 not in self.states:
            raise ValueError(f"Hirshfeld-I requires a neutral state for Z={self.atnum}.")
        state_charges = sorted(self.states)
        bounds = [(float(state_charges[0]), float(state_charges[-1]))]
        super().__init__(iatom, np.asarray(center, dtype=float), np.array([charge], float), bounds)
        _ = self.interpolation_info

    @property
    def charge(self):
        """Current fractional pro-atom charge."""
        return float(self.pars[0])

    @property
    def population(self):
        """Integrated population of the interpolated pro-atom."""
        return self.atnum - self.charge

    @property
    def population_derivatives(self):
        """Derivative of population with respect to fractional charge."""
        return np.array([-1.0])

    @property
    def interpolation_info(self):
        """Return lower/upper integer charges and upper-state weight."""
        charge = self.charge
        minimum, maximum = self.bounds[0]
        tolerance = 1.0e-10
        if charge < minimum - tolerance or charge > maximum + tolerance:
            raise ValueError(
                f"Hirshfeld-I charge {charge:+.8f} for Z={self.atnum} is outside "
                f"the available range [{minimum:+.0f}, {maximum:+.0f}]."
            )
        charge = float(np.clip(charge, minimum, maximum))
        nearest = round(charge)
        if abs(charge - nearest) < tolerance:
            if nearest not in self.states:
                raise ValueError(f"Missing Hirshfeld-I state Z={self.atnum}, charge={nearest:+d}.")
            return nearest, nearest, 0.0
        lower = int(np.floor(charge))
        upper = lower + 1
        missing = [
            state_charge for state_charge in (lower, upper) if state_charge not in self.states
        ]
        if missing:
            labels = ", ".join(f"{state_charge:+d}" for state_charge in missing)
            raise ValueError(f"Missing Hirshfeld-I state(s) {labels} for Z={self.atnum}.")
        return lower, upper, charge - lower

    def _compute_state(self, charge, points, cache=None):
        """Evaluate one fixed integer-charge density."""
        points_id = id(points) if cache is None else cache.identity(points)

        def compute():
            exponents, populations = self.states[charge]
            if len(exponents) == 0:
                return np.zeros(len(points))
            distances2 = np.einsum("ij,ij->i", points - self.center, points - self.center)
            values = populations[:, None] * (exponents[:, None] / np.pi) ** 1.5
            return np.sum(values * np.exp(-exponents[:, None] * distances2[None, :]), axis=0)

        return compute_cached(
            cache,
            until="forever",
            key=("hirshfeld-i-state", self.iatom, charge, points_id),
            func=compute,
        )

    def compute(self, points, cache=None):
        """Evaluate the current fractional-charge pro-atom density."""
        lower, upper, fraction = self.interpolation_info
        result = self._compute_state(lower, points, cache).copy()
        if upper != lower:
            result *= 1.0 - fraction
            result += fraction * self._compute_state(upper, points, cache)
        return result

    def compute_derivatives(self, points, cache=None):
        """Evaluate the piecewise derivative with respect to charge."""
        lower, upper, _ = self.interpolation_info
        if lower == upper:
            if lower < max(self.states) and lower + 1 in self.states:
                upper = lower + 1
            elif lower > min(self.states) and lower - 1 in self.states:
                lower -= 1
            else:
                return np.zeros((1, len(points)))
        derivative = self._compute_state(upper, points, cache)
        derivative = derivative - self._compute_state(lower, points, cache)
        return derivative[None, :]

    def get_cutoff_radius(self, density_cutoff):
        """Return a cutoff covering the currently interpolated charge states."""
        if density_cutoff <= 0.0:
            return np.inf
        lower, upper, _ = self.interpolation_info
        active_charges = (lower,) if lower == upper else (lower, upper)
        primitive_count = sum(len(self.states[charge][0]) for charge in active_charges)
        threshold = density_cutoff / max(primitive_count, 1)
        radius = 0.0
        for charge in active_charges:
            exponents, populations = self.states[charge]
            for exponent, population in zip(exponents, populations, strict=True):
                prefactor = population * (exponent / np.pi) ** 1.5
                if prefactor > threshold:
                    radius = max(radius, np.sqrt(np.log(prefactor / threshold) / exponent))
        return radius


class GaussianHirshfeldIProModel(ProModel):
    """Piecewise-linear charged-state model for iterative Hirshfeld partitioning."""

    @classmethod
    def from_geometry(cls, atnums, atcoords, basis=None):
        """Construct neutral initial pro-atoms from a charge-state library."""
        state_basis = load_proatom_states(basis)
        functions = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            atnum = int(atnum)
            if atnum not in state_basis:
                available = ", ".join(str(number) for number in sorted(state_basis))
                raise NotImplementedError(
                    f"No Hirshfeld-I states are available for atomic number {atnum}. "
                    f"Available atomic numbers: {available}."
                )
            functions.append(
                InterpolatedGaussianFunction(iatom, atnum, atcoord, 0.0, state_basis[atnum])
            )
        return cls(np.asarray(atnums), np.asarray(atcoords), functions)

    def to_dict(self):
        """Return a reconstructible charged-state model and convergence history."""
        result = super().to_dict()
        result["method"] = np.array("HIRSHFELD-I")
        result["state_counts"] = np.array(
            [len(function.states) for function in self.fns], dtype=int
        )
        result["state_charges"] = np.array(
            [charge for function in self.fns for charge in sorted(function.states)], dtype=int
        )
        result["state_nprimitives"] = np.array(
            [
                len(function.states[charge][0])
                for function in self.fns
                for charge in sorted(function.states)
            ],
            dtype=int,
        )
        result["state_exponents"] = np.concatenate(
            [
                function.states[charge][0]
                for function in self.fns
                for charge in sorted(function.states)
            ]
        )
        result["state_populations"] = np.concatenate(
            [
                function.states[charge][1]
                for function in self.fns
                for charge in sorted(function.states)
            ]
        )
        history = getattr(self, "charge_history", [self.charges.copy()])
        result["charge_history"] = np.asarray(history)
        result["iterations"] = np.array(max(len(history) - 1, 0))
        result["max_charge_change"] = np.array(getattr(self, "max_charge_change", 0.0))
        return result

    @classmethod
    def from_dict(cls, data):
        """Reconstruct a charged-state model from stored arrays."""
        if str(data["class"]) != "GaussianHirshfeldIProModel":
            raise TypeError("Expected a GaussianHirshfeldIProModel dictionary.")
        atnums = np.asarray(data["atnums"])
        atcoords = np.asarray(data["atcoords"])
        charges = np.asarray(data["propars"], dtype=float)
        state_counts = np.asarray(data["state_counts"], dtype=int)
        state_charges = np.asarray(data["state_charges"], dtype=int)
        primitive_counts = np.asarray(data["state_nprimitives"], dtype=int)
        exponents = np.asarray(data["state_exponents"], dtype=float)
        populations = np.asarray(data["state_populations"], dtype=float)
        functions = []
        istate = 0
        iprimitive = 0
        for iatom, (atnum, atcoord, charge, state_count) in enumerate(
            zip(atnums, atcoords, charges, state_counts, strict=True)
        ):
            states = {}
            for _ in range(state_count):
                state_charge = int(state_charges[istate])
                nprimitive = int(primitive_counts[istate])
                states[state_charge] = (
                    exponents[iprimitive : iprimitive + nprimitive],
                    populations[iprimitive : iprimitive + nprimitive],
                )
                istate += 1
                iprimitive += nprimitive
            functions.append(InterpolatedGaussianFunction(iatom, atnum, atcoord, charge, states))
        if istate != len(state_charges) or iprimitive != len(exponents):
            raise ValueError("Stored Hirshfeld-I state arrays have inconsistent lengths.")
        model = cls(atnums, atcoords, functions)
        if "charge_history" in data:
            model.charge_history = np.asarray(data["charge_history"]).tolist()
            model.max_charge_change = float(data["max_charge_change"])
        return model


def optimize_hirshfeld_i(
    pro_model,
    grid,
    density,
    threshold=1.0e-8,
    maxiter=1000,
    density_cutoff=1.0e-10,
    cache=None,
    mixing=0.5,
):
    """Iterate pro-atom charges until they match the stockholder AIM charges.

    Full fixed-point steps are used initially. If consecutive charge updates
    alternate, all later updates are under-relaxed by ``mixing``.
    """
    if not 0.0 < mixing <= 1.0:
        raise ValueError("Hirshfeld-I mixing must be greater than zero and at most one.")

    def active_states():
        return tuple(function.interpolation_info[:2] for function in pro_model.fns)

    def build_localgrids():
        return [
            grid.get_localgrid(function.center, function.get_cutoff_radius(density_cutoff))
            for function in pro_model.fns
        ]

    current_states = active_states()
    localgrids = build_localgrids()
    pro_model.charge_history = [pro_model.charges.copy()]
    previous_step = None
    use_mixing = False
    print("Hirshfeld-I iterations")
    print("#Iter  max charge change  sum charges")
    print("-----  -----------------  -----------")
    for iteration in range(1, maxiter + 1):
        promolecule = pro_model.compute_density(grid, localgrids, cache)
        sick = (density < 1.0e-15) | (promolecule < 1.0e-15)
        ratio = np.divide(density, promolecule, out=np.zeros_like(density), where=~sick)
        raw_charges = np.empty(pro_model.natom)
        for iatom, (function, localgrid) in enumerate(zip(pro_model.fns, localgrids, strict=True)):
            proatom = function.compute(localgrid.points, cache)
            population = localgrid.integrate(proatom, ratio[localgrid.indices])
            raw_charges[iatom] = pro_model.atnums[iatom] - population
            old_charge = function.pars[0]
            function.pars[0] = raw_charges[iatom]
            try:
                _ = function.interpolation_info
            finally:
                function.pars[0] = old_charge
        old_charges = pro_model.charges.copy()
        raw_step = raw_charges - old_charges
        if previous_step is not None and np.dot(raw_step, previous_step) < 0.0:
            use_mixing = True
        new_charges = old_charges + (mixing * raw_step if use_mixing else raw_step)
        previous_step = new_charges - old_charges
        pro_model.assign_pars(new_charges)
        change = float(np.max(np.abs(new_charges - old_charges)))
        pro_model.charge_history.append(new_charges.copy())
        pro_model.max_charge_change = change
        print(f"{iteration:5d}  {change:17.10e}  {new_charges.sum():+11.4e}")
        new_states = active_states()
        if new_states != current_states:
            localgrids = build_localgrids()
            current_states = new_states
        if change < threshold:
            return pro_model, localgrids
    raise RuntimeError(
        f"Hirshfeld-I did not converge in {maxiter} iterations; "
        f"last maximum charge change was {pro_model.max_charge_change:.3e}."
    )
