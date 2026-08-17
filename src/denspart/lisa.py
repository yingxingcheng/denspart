# DensPart performs Atoms-in-molecules density partitioning.
# Copyright (C) 2011-2020 The DensPart Development Team
#
# This file is part of DensPart.
#
# DensPart is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DensPart is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
# pylint: disable=too-many-lines
"""Linear approximation of Iterative Stockholder Analysis (LISA) partitioning scheme."""

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

from .cache import compute_cached
from .vh import BasisFunction, ProModel

__all__ = ["GaussianFunction", "LISAProModel", "load_lisa_basis"]


# The default Gaussian LISA basis distributed with HORTON-Part. Each entry contains
# (orders, exponents, initial populations). Only order-two (Gaussian) functions are
# supported by this implementation.
DEFAULT_LISA_BASIS = {
    1: ([2] * 4, [5.672, 1.505, 0.5308, 0.2204], [0.0429, 0.2639, 0.479, 0.2127]),
    3: (
        [2] * 6,
        [25.5115, 7.3253, 2.6324, 0.9655, 0.0965, 0.0311],
        [0.1993, 0.7643, 0.8928, 0.0898, 1.0111, 0.0250],
    ),
    5: (
        [2] * 6,
        [98.2299, 27.7169, 9.7959, 0.5004, 0.1942, 0.0618],
        [0.1356, 0.6428, 1.0597, 1.9693, 1.1509, 0.0412],
    ),
    6: (
        [2] * 6,
        [148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511],
        [0.1330, 0.5955, 1.0749, 0.0202, 2.7117, 1.4779],
    ),
    7: (
        [2] * 6,
        [178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857],
        [0.1627, 0.6567, 0.9993, 2.3257, 1.8949, 0.9479],
    ),
    8: (
        [2] * 6,
        [220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311],
        [0.1869, 0.6576, 0.9751, 3.0657, 2.5622, 0.5528],
    ),
    9: (
        [2] * 6,
        [232.2846, 73.1726, 30.0344, 2.4199, 1.0096, 0.3263],
        [0.2326, 0.7623, 0.8161, 2.9602, 3.3411, 0.8988],
    ),
    14: (
        [2] * 9,
        [366.5112, 104.3665, 15.5123, 9.5104, 7.8724, 5.3849, 3.7020, 0.3241, 0.1076],
        [0.5063, 1.1758, 0.0, 1.7484, 0.4014, 2.5315, 3.0395, 3.5767, 1.0358],
    ),
    # P and Ga were fitted to PBE/6-311+G(d,p) isolated-atom densities for
    # charge states -2 through +2 using the NLIS construction workflow.
    15: (
        [2] * 15,
        [
            1607.2229418674185,
            598.7534477660298,
            352.9197501706697,
            137.23541684996133,
            116.68041750722449,
            10.954576926310706,
            5.2934983254277235,
            0.5285489118579311,
            0.41845637409883046,
            0.31928900095453866,
            0.2426499999113163,
            0.18226668509272986,
            0.14713016647746877,
            0.07831066990016262,
            0.0558279870746359,
        ],
        [
            0.20019748266140913,
            0.7305493542807111,
            1.6195302047400812,
            2.582731315062559,
            3.27947730585005,
            12.523480985928483,
            25.653790383403543,
            4.398840376523374,
            12.60968096006409,
            3.910065872633923,
            0.5025525027028681,
            2.1803417837265986,
            1.1567418165406516,
            1.0915643960313925,
            2.5604390827631702,
        ],
    ),
    16: (
        [2] * 9,
        [528.7272, 147.5558, 17.6378, 17.5077, 15.1251, 7.1494, 0.5499, 0.2713, 0.1013],
        [0.4472, 1.1959, 0.0, 0.0, 1.4710, 6.0430, 4.2959, 2.2890, 0.2565],
    ),
    17: (
        [2] * 9,
        [622.3137, 180.7931, 98.9482, 69.1275, 20.2219, 8.9831, 0.6418, 0.3052, 0.1370],
        [0.4127, 1.1066, 0.1210, 0.0, 0.9133, 6.5025, 5.5666, 2.0125, 0.3716],
    ),
    31: (
        [2] * 14,
        [
            2839.5670843211146,
            652.5511845364244,
            47.428665267763144,
            3.809451060293093,
            1.7232957138027196,
            0.5678201792203799,
            0.4599590286913162,
            0.2837631948398335,
            0.1787137705455484,
            0.13391369441817216,
            0.06507729050623332,
            0.057310497864053414,
            0.012643195913354463,
            0.011367366677815207,
        ],
        [
            1.6395013595401968,
            6.316991269713919,
            33.851422801684365,
            65.81543566199859,
            24.371425334863154,
            3.9310650306779853,
            6.977798515888602,
            2.23081035526595,
            6.043093534994724,
            0.11793563298968199,
            0.32596952206425367,
            0.8930262453900715,
            0.4243473476327084,
            2.061159076301232,
        ],
    ),
    35: (
        [2] * 12,
        [
            1027.3862,
            84.3671,
            67.8966,
            64.9399,
            30.7992,
            6.4459,
            5.3029,
            4.4950,
            2.6361,
            0.7183,
            0.3682,
            0.1390,
        ],
        [1.4011, 0.0, 0.0, 6.6184, 0.0, 0.0, 16.7407, 0.0, 1.0632, 3.3418, 5.0013, 0.7444],
    ),
}


def load_lisa_basis(source=None):
    """Load and validate a Gaussian LISA basis.

    ``source`` may be a path, a mapping, or ``None`` for the built-in HORTON-Part
    basis. Both the HORTON-Part ``{Z: [orders, exponents, initials]}`` layout and
    the versioned ``{"format": "denspart-lisa-basis-v1", "elements": ...}``
    layout are accepted. Initial populations are normalized to the atomic number.
    """
    if source is None:
        raw_basis = DEFAULT_LISA_BASIS
    elif isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf8") as f:
            raw_basis = json.load(f)
    else:
        raw_basis = source

    if not isinstance(raw_basis, dict):
        raise TypeError("A LISA basis must be a mapping or a path to a JSON mapping.")
    if "elements" in raw_basis:
        if raw_basis.get("format") != "denspart-lisa-basis-v1":
            raise ValueError("Unsupported versioned LISA basis format.")
        raw_basis = raw_basis["elements"]

    basis = {}
    for raw_atnum, values in raw_basis.items():
        try:
            atnum = int(raw_atnum)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in LISA basis: {raw_atnum!r}.") from exc
        if isinstance(values, dict):
            orders = values.get("orders", [2] * len(values.get("exponents", [])))
            exponents = values.get("exponents")
            initials = values.get("initials")
        elif len(values) == 3:
            orders, exponents, initials = values
        else:
            raise ValueError(f"Invalid LISA basis entry for atomic number {atnum}.")

        orders = np.asarray(orders, dtype=float)
        exponents = np.asarray(exponents, dtype=float)
        if initials is None:
            initials = np.ones_like(exponents)
        initials = np.asarray(initials, dtype=float)
        if orders.ndim != 1 or exponents.ndim != 1 or initials.ndim != 1:
            raise ValueError(
                f"LISA basis arrays for atomic number {atnum} must be one-dimensional."
            )
        if not (len(orders) == len(exponents) == len(initials)) or len(exponents) == 0:
            raise ValueError(f"LISA basis arrays for atomic number {atnum} have unequal lengths.")
        if not np.all(orders == 2):
            raise ValueError("DensPart LISA currently supports only order-two Gaussian functions.")
        if not np.isfinite(exponents).all() or not (exponents > 0).all():
            raise ValueError(
                f"LISA exponents for atomic number {atnum} must be finite and positive."
            )
        if not np.isfinite(initials).all() or not (initials >= 0).all() or initials.sum() <= 0:
            raise ValueError(
                f"LISA initial populations for atomic number {atnum} must be finite, "
                "nonnegative, and not all zero."
            )
        basis[atnum] = (exponents, initials * atnum / initials.sum())
    return basis


class GaussianFunction(BasisFunction):
    """Gaussian basis function for the LISA pro density.

    See BasisFunction base class for API documentation.
    """

    def __init__(self, iatom, center, pars, exponent):
        pars = np.asarray(pars, dtype=float)
        if len(pars) != 1 or not np.isfinite(pars).all() or not (pars >= 0).all():
            raise ValueError("Expecting one finite, nonnegative population parameter.")
        if not np.isfinite(exponent) or exponent <= 0:
            raise ValueError("Expecting a finite, positive Gaussian exponent.")
        self.exponent = float(exponent)
        super().__init__(iatom, np.asarray(center), pars, [(0.0, np.inf)])

    @property
    def population(self):
        return self.pars[0]

    @property
    def population_derivatives(self):
        return np.array([1.0])

    def get_cutoff_radius(self, density_cutoff):
        if density_cutoff <= 0.0:
            return np.inf
        population, exponent = self.pars[0], self.exponent
        prefactor = population * (exponent / np.pi) ** 1.5
        if prefactor <= density_cutoff:
            return 0.0
        return np.sqrt((np.log(prefactor) - np.log(density_cutoff)) / exponent)

    def _compute_dists(self, points, cache=None):
        points_id = id(points) if cache is None else cache.identity(points)
        return compute_cached(
            cache,
            until="forever",
            key=("dists", *self.center, points_id),
            func=(lambda: np.linalg.norm(points - self.center, axis=1)),
        )

    def _compute_exp(self, exponent, dists, cache=None):
        # print(exponent, np.max(dists**2), np.min(dists**2))
        dists_id = id(dists) if cache is None else cache.identity(dists)
        return compute_cached(
            cache,
            until="end-ekld",
            key=("exp", exponent, dists_id),
            func=(lambda: np.exp(-exponent * dists**2)),
        )

    def compute(self, points, cache=None):
        population, exponent = self.pars[0], self.exponent
        if exponent < 0 or population < 0:
            return np.full(len(points), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        prefactor = population * (exponent / np.pi) ** 1.5
        return prefactor * exp

    def compute_derivatives(self, points, cache=None):
        population, exponent = self.pars[0], self.exponent
        if exponent < 0 or population < 0:
            warnings.warn("exponent or population is negative!", stacklevel=1)
            exponent = -exponent if exponent < 0 else exponent
            population = -population if population < 0 else population
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        factor = (exponent / np.pi) ** 1.5
        # vector = (population * exponent**2 / 8 / np.pi) * (3 - dists * exponent)
        return np.array([factor * exp])


class LISAProModel(ProModel):
    """ProModel for LISA partitioning."""

    @classmethod
    def from_geometry(cls, atnums, atcoords, basis=None):
        """Derive a ProModel with a sensible initial guess from a molecular geometry.

        Parameters
        ----------
        atnums
            An array with atomic numbers, shape ``(natom, )``.
        atcoords
            An array with atomic coordinates, shape ``(natom, 3)``
        """
        basis = load_lisa_basis(basis)
        fns = []
        for iatom, (atnum, atcoord) in enumerate(zip(atnums, atcoords, strict=True)):
            atnum = int(atnum)
            if atnum not in basis:
                available = ", ".join(str(number) for number in sorted(basis))
                raise NotImplementedError(
                    f"No LISA basis is available for atomic number {atnum}. "
                    f"Available atomic numbers: {available}. Supply a custom basis file."
                )
            exponents, populations = basis[atnum]
            for population, exponent in zip(populations, exponents, strict=True):
                fns.append(GaussianFunction(iatom, atcoord, [population], exponent))
        return cls(atnums, atcoords, fns)

    def reduce(self, eps=1e-4):
        """Return a new ProModel in which redundant functions are merged together.

        Parameters
        ----------
        eps
            When abs(e1 - e2) < eps * (e1 + e2) / 2, were e1 and e2 are exponents,
            two functions will be merged. Also when the population of a basis function
            is lower then eps, it is removed.

        """
        pro_model = super().reduce(eps)
        # Group functions by atoms
        grouped_fns = {}
        for fn in pro_model.fns:
            grouped_fns.setdefault(fn.iatom, []).append(fn)
        # Loop over all atoms and merge where possible
        new_fns = []
        for iatom, fns in grouped_fns.items():
            pairs = [
                (index1, index2)
                for (index1, fn1), (index2, fn2) in combinations(enumerate(fns), 2)
                if abs(fn1.exponent - fn2.exponent) < eps * (fn1.exponent + fn2.exponent) / 2
            ]
            adjacency = {index: set() for index in range(len(fns))}
            for index1, index2 in pairs:
                adjacency[index1].add(index2)
                adjacency[index2].add(index1)
            clusters = []
            unseen = set(range(len(fns)))
            while unseen:
                pending = [min(unseen)]
                cluster = []
                unseen.remove(pending[0])
                while pending:
                    index = pending.pop()
                    cluster.append(index)
                    neighbors = adjacency[index] & unseen
                    unseen.difference_update(neighbors)
                    pending.extend(sorted(neighbors, reverse=True))
                clusters.append(sorted(cluster))
            for cluster in clusters:
                population = sum(fns[index].population for index in cluster)
                exponent = sum(fns[index].exponent for index in cluster) / len(cluster)
                new_fns.append(
                    GaussianFunction(iatom, pro_model.atcoords[iatom], [population], exponent)
                )
        return pro_model.__class__(pro_model.atnums, pro_model.atcoords, new_fns)

    def to_dict(self):
        """Return dictionary with additional results derived from the pro-parameters."""
        results = super().to_dict()
        results["exponents"] = np.array([fn.exponent for fn in self.fns])
        return results

    @classmethod
    def from_dict(cls, data):
        """Recreate the pro-model from a dictionary."""
        if data["class"] != "LISAProModel":
            raise TypeError("The dictionary class field should be LISAProModel.")
        fns = []
        ipar = 0
        atnums = data["atnums"]
        atcoords = data["atcoords"]
        pars = data["propars"]
        atnfns = data["atnfns"]
        stored_exponents = data.get("exponents")
        for iatom, atcoord in enumerate(atcoords):
            if stored_exponents is None:
                exponents = get_alpha(atnums[iatom])
            else:
                exponents = stored_exponents[ipar : ipar + atnfns[iatom]]
            if len(exponents) != atnfns[iatom]:
                raise ValueError(f"Wrong number of stored LISA exponents for atom {iatom}.")
            for iprim in range(atnfns[iatom]):
                fn_pars = pars[ipar]
                fns.append(GaussianFunction(iatom, atcoord, [fn_pars], exponents[iprim]))
                ipar += 1
        return cls(atnums, atcoords, fns)


def get_alpha(atnum):
    """The exponents used for primitive Gaussian functions of each element."""
    basis = load_lisa_basis()
    if int(atnum) not in basis:
        raise NotImplementedError(f"No default LISA basis for atomic number {atnum}.")
    return basis[int(atnum)][0].copy()


def get_initial_population(atnum, exponents):
    """Get initial population based on atomic number `atnum`."""
    basis = load_lisa_basis()
    atnum = int(atnum)
    exponents = np.asarray(exponents)
    if atnum in basis and np.array_equal(exponents, basis[atnum][0]):
        return basis[atnum][1].copy()
    return np.full_like(exponents, atnum / len(exponents), dtype=float)
