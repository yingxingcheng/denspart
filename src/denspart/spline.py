# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
"""Shared radial-spline pro-atoms for Hirshfeld, Hirshfeld-I, and AVH."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .cache import compute_cached
from .vh import BasisFunction, ProModel

__all__ = [
    "SplineProModel",
    "SplineStateFunction",
    "is_spline_basis",
    "load_spline_basis",
    "optimize_spline_hirshfeld_i",
]

SPLINE_PROATOM_FORMATS = frozenset(
    {"aim-proatom-spline-v1", "denspart-spline-proatom-basis-v1"}
)


def _load_mapping(source):
    """Load a JSON mapping from a path or return a supplied mapping."""
    if source is None:
        raise ValueError("An aim-proatom-spline-v1 basis file is required.")
    if isinstance(source, str | Path):
        with Path(source).open(encoding="utf8") as handle:
            return json.load(handle)
    return source


def is_spline_basis(source):
    """Return whether ``source`` uses the radial-spline library format."""
    if source is None:
        return False
    library = _load_mapping(source)
    return isinstance(library, dict) and library.get("format") in SPLINE_PROATOM_FORMATS


def load_spline_basis(source, avh_variant=None):
    """Load electron-normalized isolated-atom densities on radial grids.

    Each returned state contains a unit-integral shape.  Its electron count is
    retained separately, so all supported partitioning methods differ only in
    the outer population coefficients applied to these fixed shapes.
    """
    library = _load_mapping(source)
    if not isinstance(library, dict) or library.get("format") not in SPLINE_PROATOM_FORMATS:
        names = " or ".join(sorted(SPLINE_PROATOM_FORMATS))
        raise ValueError(f"Expected an {names} mapping.")
    elements = library.get("elements")
    if not isinstance(elements, dict) or not elements:
        raise ValueError("The spline pro-atom basis contains no elements.")

    result = {}
    for raw_atnum, element in elements.items():
        try:
            atnum = int(raw_atnum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in spline basis: {raw_atnum!r}.") from exc
        radii = np.asarray(element.get("radii"), dtype=float)
        radial_weights = np.asarray(element.get("radial_weights"), dtype=float)
        if radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all():
            raise ValueError(f"Spline radii for Z={atnum} must be a finite 1D array.")
        if radii[0] < 0.0 or not np.all(np.diff(radii) > 0.0):
            raise ValueError(f"Spline radii for Z={atnum} must be nonnegative and increasing.")
        if radial_weights.shape != radii.shape or not np.isfinite(radial_weights).all():
            raise ValueError(f"Radial weights for Z={atnum} must match its radial grid.")
        if np.any(radial_weights <= 0.0):
            raise ValueError(f"Radial weights for Z={atnum} must be positive.")

        states = []
        seen_charges = set()
        volume_weights = 4.0 * np.pi * radii**2 * radial_weights
        for state in element.get("states", []):
            raw_charge = state.get("charge")
            if not isinstance(raw_charge, int | np.integer):
                raise ValueError(f"Spline charges for Z={atnum} must be integers.")
            charge = int(raw_charge)
            if charge in seen_charges:
                raise ValueError(f"Duplicate spline charge {charge:+d} for Z={atnum}.")
            seen_charges.add(charge)
            electrons = int(state.get("electrons", atnum - charge))
            if electrons != atnum - charge or electrons < 0:
                raise ValueError(
                    f"Spline state Z={atnum}, charge={charge:+d} has an invalid electron count."
                )
            density = np.asarray(state.get("density"), dtype=float)
            if density.shape != radii.shape or not np.isfinite(density).all():
                raise ValueError(
                    f"Spline density for Z={atnum}, charge={charge:+d} must match its radial grid."
                )
            if np.any(density < 0.0):
                raise ValueError(
                    f"Spline density for Z={atnum}, charge={charge:+d} must be nonnegative."
                )
            population = float(np.dot(volume_weights, density))
            if not np.isclose(population, electrons, rtol=0.0, atol=1.0e-6):
                raise ValueError(
                    f"Spline state Z={atnum}, charge={charge:+d} integrates to "
                    f"{population:.12g}, expected {electrons}."
                )
            shape = density / electrons if electrons else np.zeros_like(density)
            states.append(
                (
                    charge,
                    electrons,
                    radii.copy(),
                    shape,
                    state.get("bound_to_electron_loss"),
                )
            )
        if not states:
            raise ValueError(f"Spline basis for Z={atnum} contains no states.")
        if 0 not in seen_charges:
            raise ValueError(f"Spline basis for Z={atnum} must contain a neutral state.")
        states.sort(key=lambda item: item[0])
        if avh_variant is not None:
            states = _select_avh_states(states, avh_variant, atnum)
        result[atnum] = [state[:4] for state in states]
    return result


def _select_avh_states(states, variant, atnum):
    """Select AVH-A/B/M states from one complete package-neutral library."""
    variant = variant.upper()
    by_charge = {state[0]: state for state in states if state[1] > 0}
    if variant == "SUPPLIED":
        return list(by_charge.values())
    if variant == "M":
        required = [0]
    elif variant == "A":
        required = list(range(-3, atnum))
    elif variant == "B":
        required = list(range(0, atnum))
        anion = by_charge.get(-1)
        if anion is not None and anion[4] is not False:
            required.insert(0, -1)
    else:
        raise ValueError("AVH variant must be 'A', 'B', 'M', or 'supplied'.")
    missing = [charge for charge in required if charge not in by_charge]
    if missing:
        labels = ", ".join(f"{charge:+d}" for charge in missing)
        raise ValueError(f"AVH-{variant} for Z={atnum} is missing required states: {labels}.")
    return [by_charge[charge] for charge in required]


class SplineStateFunction(BasisFunction):
    """One fixed, unit-integral atomic-state spline with an outer coefficient."""

    def __init__(
        self,
        iatom,
        center,
        charge,
        electrons,
        population,
        radii,
        density,
        population_scale,
    ):
        self.charge = int(charge)
        self.electrons = int(electrons)
        self.radii = np.asarray(radii, dtype=float)
        self.density = np.asarray(density, dtype=float)
        self.population_scale = float(population_scale)
        spline_radii = self.radii
        spline_density = self.density
        if spline_radii[0] > 0.0:
            spline_radii = np.concatenate(([0.0], spline_radii))
            spline_density = np.concatenate(([spline_density[0]], spline_density))
        self._spline = CubicSpline(
            spline_radii,
            spline_density,
            bc_type=((1, 0.0), "natural"),
            extrapolate=False,
        )
        super().__init__(
            iatom,
            np.asarray(center, dtype=float),
            np.array([population], dtype=float),
            [(0.0, np.inf)],
        )

    @property
    def population(self):
        """Integrated contribution of this state to its pro-atom."""
        return float(self.pars[0])

    @property
    def population_derivatives(self):
        """Derivative of the population with respect to its coefficient."""
        return np.array([1.0])

    @property
    def state_multiplier(self):
        """Coefficient multiplying the original electron-normalized density."""
        return self.population / self.electrons if self.electrons else 0.0

    def _compute_shape(self, points, cache=None):
        """Evaluate the nonnegative radial spline at Cartesian points."""
        points_id = id(points) if cache is None else cache.identity(points)

        def compute():
            distances = np.linalg.norm(points - self.center, axis=1)
            values = self._spline(distances)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(values, 0.0, np.inf)

        return compute_cached(
            cache,
            until="forever",
            key=("spline-state", self.iatom, self.charge, points_id),
            func=compute,
        )

    def compute(self, points, cache=None):
        """Evaluate the population-scaled state shape."""
        if self.population == 0.0:
            return np.zeros(len(points))
        return self.population * self._compute_shape(points, cache)

    def compute_derivatives(self, points, cache=None):
        """Evaluate the derivative with respect to the outer coefficient."""
        return self._compute_shape(points, cache)[None, :]

    def _get_cutoff_radius(self, density_cutoff, scale):
        if density_cutoff <= 0.0:
            return np.inf
        if scale <= 0.0 or not np.any(self.density):
            return 0.0
        active = np.flatnonzero(scale * self.density >= density_cutoff)
        if not len(active):
            return 0.0
        index = min(int(active[-1]) + 1, len(self.radii) - 1)
        return float(self.radii[index])

    def get_cutoff_radius(self, density_cutoff):
        """Return a cutoff valid even if this currently inactive state becomes active."""
        return self._get_cutoff_radius(density_cutoff, max(self.population_scale, self.population))

    def get_current_cutoff_radius(self, density_cutoff):
        """Return the cutoff for the current state contribution."""
        return self._get_cutoff_radius(density_cutoff, self.population)

    def get_reference_cutoff_radius(self, density_cutoff):
        """Return the cutoff for the full isolated-state population."""
        return self._get_cutoff_radius(density_cutoff, self.electrons)


class SplineProModel(ProModel):
    """Common spline model whose methods update only state populations."""

    METHODS = frozenset({"HIRSHFELD", "HIRSHFELD-I", "AVH"})

    def __init__(self, atnums, atcoords, fns, method):
        if method not in self.METHODS:
            raise ValueError(f"Unsupported spline partitioning method: {method!r}.")
        self.method = method
        super().__init__(atnums, atcoords, fns)

    @classmethod
    def from_geometry(
        cls,
        atnums,
        atcoords,
        basis=None,
        method="HIRSHFELD",
        avh_variant="supplied",
    ):
        """Construct neutral pro-atoms and the state set required by ``method``."""
        state_basis = load_spline_basis(
            basis, avh_variant=avh_variant if method == "AVH" else None
        )
        functions = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            atnum = int(atnum)
            if atnum not in state_basis:
                available = ", ".join(str(number) for number in sorted(state_basis))
                raise NotImplementedError(
                    f"No spline pro-atoms are available for atomic number {atnum}. "
                    f"Available atomic numbers: {available}."
                )
            states = state_basis[atnum]
            if method == "HIRSHFELD":
                states = [state for state in states if state[0] == 0]
            elif method == "AVH":
                states = [state for state in states if state[1] > 0]
            for charge, electrons, radii, shape in states:
                population = float(atnum) if charge == 0 else 0.0
                functions.append(
                    SplineStateFunction(
                        iatom,
                        atcoord,
                        charge,
                        electrons,
                        population,
                        radii,
                        shape,
                        atnum + 3,
                    )
                )
        return cls(np.asarray(atnums), np.asarray(atcoords), functions, method)

    def _atom_functions(self, iatom):
        return [function for function in self.fns if function.iatom == iatom]

    def compute_density(self, grid, localgrids=None, cache=None):
        """Evaluate only states with nonzero mixing coefficients."""
        density = np.zeros_like(grid.weights)
        if localgrids is None:
            for function in self.fns:
                if function.population != 0.0:
                    density += function.compute(grid.points, cache)
        else:
            for function, localgrid in zip(self.fns, localgrids, strict=True):
                if function.population != 0.0:
                    np.add.at(
                        density,
                        localgrid.indices,
                        function.compute(localgrid.points, cache),
                    )
        return density

    def compute_proatom(self, iatom, points, cache=None):
        """Evaluate only active state contributions for one atom."""
        density = np.zeros(len(points))
        for function in self._atom_functions(iatom):
            if function.population != 0.0:
                density += function.compute(points, cache)
        return density

    def interpolation_info(self, iatom, charge=None):
        """Return adjacent integer states and the upper-state mixing fraction."""
        functions = self._atom_functions(iatom)
        states = {function.charge: function for function in functions}
        if charge is None:
            charge = float(self.charges[iatom])
        minimum, maximum = min(states), max(states)
        if maximum == int(self.atnums[iatom]) - 1:
            maximum += 1  # The fully stripped, zero-density state is implicit.
        tolerance = 1.0e-10
        if charge < minimum - tolerance or charge > maximum + tolerance:
            raise ValueError(
                f"Hirshfeld-I charge {charge:+.8f} for Z={self.atnums[iatom]} is outside "
                f"the available range [{minimum:+d}, {maximum:+d}]."
            )
        charge = float(np.clip(charge, minimum, maximum))
        nearest = round(charge)
        if abs(charge - nearest) < tolerance:
            if nearest not in states and nearest != int(self.atnums[iatom]):
                raise ValueError(
                    f"Missing Hirshfeld-I state Z={self.atnums[iatom]}, charge={nearest:+d}."
                )
            return nearest, nearest, 0.0
        lower = int(np.floor(charge))
        upper = lower + 1
        missing = [
            state
            for state in (lower, upper)
            if state not in states and state != int(self.atnums[iatom])
        ]
        if missing:
            labels = ", ".join(f"{state:+d}" for state in missing)
            raise ValueError(f"Missing Hirshfeld-I state(s) {labels} for Z={self.atnums[iatom]}.")
        return lower, upper, charge - lower

    def set_hirshfeld_i_charges(self, charges):
        """Represent fractional charges by adjacent-state population coefficients."""
        charges = np.asarray(charges, dtype=float)
        if charges.shape != (self.natom,):
            raise ValueError("Hirshfeld-I charges must contain one value per atom.")
        for iatom, charge in enumerate(charges):
            functions = self._atom_functions(iatom)
            states = {function.charge: function for function in functions}
            lower, upper, fraction = self.interpolation_info(iatom, charge)
            for function in functions:
                function.pars[0] = 0.0
            if lower in states:
                states[lower].pars[0] = (1.0 - fraction) * states[lower].electrons
            if upper != lower and upper in states:
                states[upper].pars[0] = fraction * states[upper].electrons

    def get_cutoff_radii(self, density_cutoff):
        """Estimate atomwise radii from current nonzero state contributions."""
        radii = np.zeros(self.natom, dtype=float)
        for function in self.fns:
            radii[function.iatom] = max(
                radii[function.iatom], function.get_current_cutoff_radius(density_cutoff)
            )
        return radii

    def to_dict(self):
        """Return a reconstructible spline model and its coefficient metadata."""
        result = super().to_dict()
        result["method"] = np.array(self.method)
        result["state_charges"] = np.array([function.charge for function in self.fns], dtype=int)
        result["state_electrons"] = np.array(
            [function.electrons for function in self.fns], dtype=int
        )
        result["state_multipliers"] = np.array([function.state_multiplier for function in self.fns])
        result["radial_counts"] = np.array(
            [len(function.radii) for function in self.fns], dtype=int
        )
        result["state_radii"] = np.concatenate([function.radii for function in self.fns])
        result["state_densities"] = np.concatenate([function.density for function in self.fns])
        result["population_scales"] = np.array([function.population_scale for function in self.fns])
        history = getattr(self, "charge_history", [self.charges.copy()])
        result["charge_history"] = np.asarray(history)
        result["iterations"] = np.array(max(len(history) - 1, 0))
        result["max_charge_change"] = np.array(getattr(self, "max_charge_change", 0.0))
        result["optimizer_iterations"] = np.array(getattr(self, "optimizer_iterations", 0))
        result["optimizer_objective"] = np.array(getattr(self, "optimizer_objective", np.nan))
        return result

    @classmethod
    def from_dict(cls, data):
        """Reconstruct a spline model from stored radial arrays."""
        if str(data["class"]) != "SplineProModel":
            raise TypeError("Expected a SplineProModel dictionary.")
        atnums = np.asarray(data["atnums"])
        atcoords = np.asarray(data["atcoords"])
        atnfns = np.asarray(data["atnfns"], dtype=int)
        populations = np.asarray(data["propars"], dtype=float)
        charges = np.asarray(data["state_charges"], dtype=int)
        electrons = np.asarray(data["state_electrons"], dtype=int)
        radial_counts = np.asarray(data["radial_counts"], dtype=int)
        radii = np.asarray(data["state_radii"], dtype=float)
        densities = np.asarray(data["state_densities"], dtype=float)
        scales = np.asarray(data["population_scales"], dtype=float)
        functions = []
        ifunction = 0
        iradial = 0
        for iatom, (atcoord, nfunction) in enumerate(zip(atcoords, atnfns, strict=True)):
            for _ in range(nfunction):
                count = int(radial_counts[ifunction])
                functions.append(
                    SplineStateFunction(
                        iatom,
                        atcoord,
                        charges[ifunction],
                        electrons[ifunction],
                        populations[ifunction],
                        radii[iradial : iradial + count],
                        densities[iradial : iradial + count],
                        scales[ifunction],
                    )
                )
                ifunction += 1
                iradial += count
        if ifunction != len(populations) or iradial != len(radii):
            raise ValueError("Stored spline state arrays have inconsistent lengths.")
        model = cls(atnums, atcoords, functions, str(data["method"]))
        if "charge_history" in data:
            model.charge_history = np.asarray(data["charge_history"]).tolist()
            model.max_charge_change = float(data["max_charge_change"])
        if "optimizer_iterations" in data:
            model.optimizer_iterations = int(data["optimizer_iterations"])
            model.optimizer_objective = float(data["optimizer_objective"])
        return model


def optimize_spline_hirshfeld_i(
    pro_model,
    grid,
    density,
    threshold=1.0e-8,
    maxiter=1000,
    density_cutoff=1.0e-10,
    cache=None,
    mixing=0.5,
):
    """Update only adjacent-state coefficients until stockholder charges converge."""
    if pro_model.method != "HIRSHFELD-I":
        raise ValueError("Spline Hirshfeld-I optimization requires a HIRSHFELD-I model.")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("Hirshfeld-I mixing must be greater than zero and at most one.")

    def active_states():
        return tuple(pro_model.interpolation_info(iatom)[:2] for iatom in range(pro_model.natom))

    def build_localgrids():
        atom_radii = np.zeros(pro_model.natom)
        for function in pro_model.fns:
            active = function.charge in current_states[function.iatom]
            if active:
                atom_radii[function.iatom] = max(
                    atom_radii[function.iatom],
                    function.get_reference_cutoff_radius(density_cutoff),
                )
        atom_grids = [
            grid.get_localgrid(center, radius)
            for center, radius in zip(pro_model.atcoords, atom_radii, strict=True)
        ]
        return [atom_grids[function.iatom] for function in pro_model.fns]

    current_states = active_states()
    localgrids = build_localgrids()
    pro_model.charge_history = [pro_model.charges.copy()]
    previous_step = None
    use_mixing = False
    print("Spline Hirshfeld-I iterations")
    print("#Iter  max charge change  sum charges")
    print("-----  -----------------  -----------")
    for iteration in range(1, maxiter + 1):
        promolecule = pro_model.compute_density(grid, localgrids, cache)
        sick = (density < 1.0e-15) | (promolecule < 1.0e-15)
        ratio = np.divide(density, promolecule, out=np.zeros_like(density), where=~sick)
        raw_charges = np.empty(pro_model.natom)
        atom_radii = pro_model.get_cutoff_radii(density_cutoff)
        for iatom, (center, radius) in enumerate(zip(pro_model.atcoords, atom_radii, strict=True)):
            localgrid = grid.get_localgrid(center, radius)
            proatom = pro_model.compute_proatom(iatom, localgrid.points, cache)
            population = localgrid.integrate(proatom, ratio[localgrid.indices])
            raw_charges[iatom] = pro_model.atnums[iatom] - population
            pro_model.interpolation_info(iatom, raw_charges[iatom])
        old_charges = pro_model.charges.copy()
        raw_step = raw_charges - old_charges
        if previous_step is not None and np.dot(raw_step, previous_step) < 0.0:
            use_mixing = True
        new_charges = old_charges + (mixing * raw_step if use_mixing else raw_step)
        previous_step = new_charges - old_charges
        pro_model.set_hirshfeld_i_charges(new_charges)
        change = float(np.max(np.abs(new_charges - old_charges)))
        pro_model.charge_history.append(new_charges.copy())
        pro_model.max_charge_change = change
        print(f"{iteration:5d}  {change:17.10e}  {new_charges.sum():+11.4e}")
        new_states = active_states()
        if new_states != current_states:
            current_states = new_states
            localgrids = build_localgrids()
        if change < threshold:
            return pro_model, localgrids
    raise RuntimeError(
        f"Spline Hirshfeld-I did not converge in {maxiter} iterations; "
        f"last maximum charge change was {pro_model.max_charge_change:.3e}."
    )
