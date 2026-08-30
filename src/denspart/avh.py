# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
"""Additive Variational Hirshfeld with contracted Gaussian state densities."""

from functools import partial

import numpy as np
from scipy.optimize import Bounds, minimize

from .cache import compute_cached
from .hirshfeld import _load_library, _validate_state_primitives
from .vh import BasisFunction, ProModel, ekld

__all__ = [
    "AVHProModel",
    "ContractedGaussianFunction",
    "load_avh_basis",
    "optimize_avh_pro_model",
]


def load_avh_basis(source):
    """Load normalized contracted state shapes from an AVH basis library."""
    _, elements = _load_library(
        source, {"aim-avh-gaussian-v1", "denspart-avh-basis-v1"}
    )
    result = {}
    for raw_atnum, element in elements.items():
        try:
            atnum = int(raw_atnum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in AVH basis: {raw_atnum!r}.") from exc
        states = []
        seen_charges = set()
        for state in element.get("states", []):
            raw_charge = state.get("charge")
            if not isinstance(raw_charge, int | np.integer):
                raise ValueError(f"AVH charge states for atomic number {atnum} must be integers.")
            charge = int(raw_charge)
            if charge in seen_charges:
                raise ValueError(f"Duplicate AVH charge {charge:+d} for atomic number {atnum}.")
            seen_charges.add(charge)
            electrons = int(state.get("electrons", atnum - charge))
            if electrons != atnum - charge or electrons <= 0:
                raise ValueError(
                    f"AVH state Z={atnum}, charge={charge:+d} has an invalid electron count."
                )
            exponents, coefficients = _validate_state_primitives(
                atnum,
                charge,
                state,
                key="shape_primitives",
                normalized=True,
            )
            states.append((charge, electrons, exponents, coefficients))
        if not states:
            raise ValueError(f"AVH basis for atomic number {atnum} contains no states.")
        if 0 not in seen_charges:
            raise ValueError(f"AVH basis for atomic number {atnum} must contain a neutral state.")
        result[atnum] = states
    return result


class ContractedGaussianFunction(BasisFunction):
    """One normalized, contracted atomic-state shape with an optimized population."""

    def __init__(
        self,
        iatom,
        center,
        charge,
        electrons,
        population,
        exponents,
        coefficients,
        population_scale,
    ):
        self.charge = int(charge)
        self.electrons = int(electrons)
        self.exponents = np.asarray(exponents, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.population_scale = float(population_scale)
        super().__init__(
            iatom,
            np.asarray(center, dtype=float),
            np.array([population], dtype=float),
            [(0.0, np.inf)],
        )

    @property
    def population(self):
        """Integrated contribution of this state shape to the pro-atom."""
        return float(self.pars[0])

    @property
    def population_derivatives(self):
        """Derivative of the population with respect to its outer coefficient."""
        return np.array([1.0])

    @property
    def state_multiplier(self):
        """Coefficient multiplying the original electron-normalized state density."""
        return self.population / self.electrons

    def _compute_shape(self, points, cache=None):
        """Evaluate the fixed unit-integral contracted shape."""
        points_id = id(points) if cache is None else cache.identity(points)

        def compute():
            distances2 = np.einsum("ij,ij->i", points - self.center, points - self.center)
            values = self.coefficients[:, None] * (self.exponents[:, None] / np.pi) ** 1.5
            return np.sum(values * np.exp(-self.exponents[:, None] * distances2[None, :]), axis=0)

        return compute_cached(
            cache,
            until="forever",
            key=("avh-shape", self.iatom, self.charge, points_id),
            func=compute,
        )

    def compute(self, points, cache=None):
        """Evaluate the scaled contracted state density."""
        return self.population * self._compute_shape(points, cache)

    def compute_derivatives(self, points, cache=None):
        """Evaluate the derivative with respect to the outer coefficient."""
        return self._compute_shape(points, cache)[None, :]

    def get_cutoff_radius(self, density_cutoff):
        """Return a cutoff that remains valid when a zero coefficient becomes active."""
        return self._get_cutoff_radius(density_cutoff, max(self.population_scale, self.population))

    def get_current_cutoff_radius(self, density_cutoff):
        """Return the cutoff needed by the current optimized contribution."""
        return self._get_cutoff_radius(density_cutoff, self.population)

    def _get_cutoff_radius(self, density_cutoff, scale):
        """Return a Gaussian-tail cutoff for a specified outer scale."""
        if density_cutoff <= 0.0:
            return np.inf
        if scale <= 0.0:
            return 0.0
        threshold = density_cutoff / (scale * len(self.exponents))
        radius = 0.0
        for exponent, coefficient in zip(self.exponents, self.coefficients, strict=True):
            prefactor = coefficient * (exponent / np.pi) ** 1.5
            if prefactor > threshold:
                radius = max(radius, np.sqrt(np.log(prefactor / threshold) / exponent))
        return radius


class AVHProModel(ProModel):
    """Positive linear expansion in normalized isolated atomic-state shapes."""

    @classmethod
    def from_geometry(cls, atnums, atcoords, basis=None):
        """Construct an AVH model initialized with neutral reference densities."""
        state_basis = load_avh_basis(basis)
        functions = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            atnum = int(atnum)
            if atnum not in state_basis:
                available = ", ".join(str(number) for number in sorted(state_basis))
                raise NotImplementedError(
                    f"No AVH basis is available for atomic number {atnum}. "
                    f"Available atomic numbers: {available}."
                )
            for charge, electrons, exponents, coefficients in state_basis[atnum]:
                initial_population = float(atnum) if charge == 0 else 0.0
                functions.append(
                    ContractedGaussianFunction(
                        iatom,
                        atcoord,
                        charge,
                        electrons,
                        initial_population,
                        exponents,
                        coefficients,
                        atnum + 2,
                    )
                )
        return cls(np.asarray(atnums), np.asarray(atcoords), functions)

    def to_dict(self):
        """Return a reconstructible AVH model with contracted-state metadata."""
        result = super().to_dict()
        result["method"] = np.array("AVH")
        result["state_charges"] = np.array([function.charge for function in self.fns], dtype=int)
        result["state_electrons"] = np.array(
            [function.electrons for function in self.fns], dtype=int
        )
        result["state_multipliers"] = np.array([function.state_multiplier for function in self.fns])
        result["primitive_counts"] = np.array(
            [len(function.exponents) for function in self.fns], dtype=int
        )
        result["state_exponents"] = np.concatenate([function.exponents for function in self.fns])
        result["shape_coefficients"] = np.concatenate(
            [function.coefficients for function in self.fns]
        )
        result["optimizer_iterations"] = np.array(getattr(self, "optimizer_iterations", 0))
        result["optimizer_objective"] = np.array(getattr(self, "optimizer_objective", np.nan))
        return result

    def get_cutoff_radii(self, density_cutoff):
        """Estimate atomwise cutoffs from only the current state populations."""
        radii = np.zeros(self.natom, dtype=float)
        for function in self.fns:
            radii[function.iatom] = max(
                radii[function.iatom], function.get_current_cutoff_radius(density_cutoff)
            )
        return radii

    @classmethod
    def from_dict(cls, data):
        """Reconstruct an AVH model from stored arrays."""
        if str(data["class"]) != "AVHProModel":
            raise TypeError("Expected an AVHProModel dictionary.")
        atnums = np.asarray(data["atnums"])
        atcoords = np.asarray(data["atcoords"])
        atnfns = np.asarray(data["atnfns"], dtype=int)
        populations = np.asarray(data["propars"], dtype=float)
        state_charges = np.asarray(data["state_charges"], dtype=int)
        state_electrons = np.asarray(data["state_electrons"], dtype=int)
        primitive_counts = np.asarray(data["primitive_counts"], dtype=int)
        exponents = np.asarray(data["state_exponents"], dtype=float)
        coefficients = np.asarray(data["shape_coefficients"], dtype=float)
        functions = []
        ifunction = 0
        iprimitive = 0
        for iatom, (atnum, atcoord, nfunction) in enumerate(
            zip(atnums, atcoords, atnfns, strict=True)
        ):
            for _ in range(nfunction):
                nprimitive = int(primitive_counts[ifunction])
                functions.append(
                    ContractedGaussianFunction(
                        iatom,
                        atcoord,
                        state_charges[ifunction],
                        state_electrons[ifunction],
                        populations[ifunction],
                        exponents[iprimitive : iprimitive + nprimitive],
                        coefficients[iprimitive : iprimitive + nprimitive],
                        int(atnum) + 2,
                    )
                )
                ifunction += 1
                iprimitive += nprimitive
        if ifunction != len(populations) or iprimitive != len(exponents):
            raise ValueError("Stored AVH state arrays have inconsistent lengths.")
        model = cls(atnums, atcoords, functions)
        if "optimizer_iterations" in data:
            model.optimizer_iterations = int(data["optimizer_iterations"])
            model.optimizer_objective = float(data["optimizer_objective"])
        return model


def optimize_avh_pro_model(
    pro_model,
    grid,
    density,
    gtol=1.0e-8,
    maxiter=1000,
    density_cutoff=1.0e-10,
    cache=None,
):
    """Minimize the AVH extended KL objective with nonnegative SLSQP coefficients."""
    print("Building AVH local grids")
    localgrids = [
        grid.get_localgrid(function.center, function.get_cutoff_radius(density_cutoff))
        for function in pro_model.fns
    ]
    population = np.einsum("i,i", grid.weights, density)
    parameters = np.concatenate([function.pars for function in pro_model.fns])
    cost_gradient = partial(
        ekld,
        grid=grid,
        density=density,
        pro_model=pro_model,
        localgrids=localgrids,
        pop=population,
        cache=cache,
    )
    result = minimize(
        cost_gradient,
        parameters,
        method="SLSQP",
        jac=True,
        bounds=Bounds(np.zeros_like(parameters), np.full_like(parameters, np.inf)),
        options={"ftol": gtol, "maxiter": maxiter, "disp": True},
    )
    if not result.success:
        raise RuntimeError(f"AVH convergence failure: {result.message}")
    optimized_population = float(np.sum(result.x))
    if optimized_population <= 0.0:
        raise RuntimeError("AVH convergence failure: optimized population is not positive.")
    parameters = result.x * (population / optimized_population)
    objective, _ = cost_gradient(parameters)
    pro_model.assign_pars(parameters)
    pro_model.optimizer_iterations = int(result.nit)
    pro_model.optimizer_objective = float(objective)
    print(f"Total charge:       {pro_model.atnums.sum() - population:20.7e}")
    print(f"Sum atomic charges: {pro_model.charges.sum():20.7e}")
    if cache is not None:
        cache.clear()
    print("Building final AVH local grids")
    final_localgrids = [
        grid.get_localgrid(function.center, function.get_current_cutoff_radius(density_cutoff))
        for function in pro_model.fns
    ]
    return pro_model, final_localgrids
