#!/usr/bin/env python3
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
"""Main command-line interface to denspart."""

import argparse

import numpy as np
from grid.basegrid import Grid
from grid.periodicgrid import PeriodicGrid

from .avh import AVHProModel, optimize_avh_pro_model
from .cache import ComputeCache
from .hirshfeld import GaussianHirshfeldProModel
from .hirshfeld_i import GaussianHirshfeldIProModel, optimize_hirshfeld_i
from .lisa import LISAProModel
from .mbis import MBISProModel
from .properties import compute_multipole_moments, compute_radial_moments
from .spline import (
    SplineProModel,
    is_spline_basis,
    optimize_spline_hirshfeld_i,
)
from .vh import optimize_pro_model, optimize_reduce_pro_model

__all__ = ["main"]


def main(args=None):
    """Partitioning command-line interface."""
    args = parse_args(args)
    nshell_map = parse_nshell_arg(args.nshell)
    data = np.load(args.in_npz)
    if "cellvecs" not in data or data["cellvecs"].size == 0:
        grid = Grid(data["points"], data["weights"])
    else:
        print("Using periodic grid")
        grid = PeriodicGrid(data["points"], data["weights"], data["cellvecs"], wrap=True)
    density = data["density"]
    if args.method == "LISA":
        print("LISA partitioning --")
        if args.nshell:
            raise ValueError("--nshell applies only to MBIS, not LISA.")
        pro_model_init = LISAProModel.from_geometry(
            data["atnums"], data["atcoords"], basis=args.lisa_basis
        )
        if args.proatom_basis is not None:
            raise ValueError("--proatom-basis applies only to HIRSHFELD, not LISA.")
        if args.avh_basis is not None:
            raise ValueError("--avh-basis applies only to AVH, not LISA.")
    elif args.method == "HIRSHFELD":
        representation = "spline" if is_spline_basis(args.proatom_basis) else "Gaussian"
        print(f"{representation}-reference Hirshfeld partitioning --")
        if args.nshell:
            raise ValueError("--nshell applies only to MBIS, not HIRSHFELD.")
        if args.lisa_basis is not None:
            raise ValueError("--lisa-basis applies only to LISA, not HIRSHFELD.")
        if representation == "spline":
            pro_model_init = SplineProModel.from_geometry(
                data["atnums"],
                data["atcoords"],
                basis=args.proatom_basis,
                method=args.method,
            )
        else:
            pro_model_init = GaussianHirshfeldProModel.from_geometry(
                data["atnums"], data["atcoords"], basis=args.proatom_basis
            )
        if args.avh_basis is not None:
            raise ValueError("--avh-basis applies only to AVH, not HIRSHFELD.")
    elif args.method == "HIRSHFELD-I":
        representation = "spline" if is_spline_basis(args.proatom_basis) else "Gaussian"
        print(f"{representation}-reference iterative Hirshfeld partitioning --")
        if args.nshell:
            raise ValueError("--nshell applies only to MBIS, not HIRSHFELD-I.")
        if args.lisa_basis is not None:
            raise ValueError("--lisa-basis applies only to LISA, not HIRSHFELD-I.")
        if args.avh_basis is not None:
            raise ValueError("--avh-basis applies only to AVH, not HIRSHFELD-I.")
        if representation == "spline":
            pro_model_init = SplineProModel.from_geometry(
                data["atnums"],
                data["atcoords"],
                basis=args.proatom_basis,
                method=args.method,
            )
        else:
            pro_model_init = GaussianHirshfeldIProModel.from_geometry(
                data["atnums"], data["atcoords"], basis=args.proatom_basis
            )
    elif args.method == "AVH":
        representation = "spline" if is_spline_basis(args.avh_basis) else "Gaussian"
        print(f"{representation} Additive Variational Hirshfeld partitioning --")
        if args.nshell:
            raise ValueError("--nshell applies only to MBIS, not AVH.")
        if args.lisa_basis is not None:
            raise ValueError("--lisa-basis applies only to LISA, not AVH.")
        if args.proatom_basis is not None:
            raise ValueError("--proatom-basis applies only to HIRSHFELD methods, not AVH.")
        if representation == "spline":
            pro_model_init = SplineProModel.from_geometry(
                data["atnums"],
                data["atcoords"],
                basis=args.avh_basis,
                method=args.method,
                avh_variant=args.avh_variant,
            )
        else:
            pro_model_init = AVHProModel.from_geometry(
                data["atnums"], data["atcoords"], basis=args.avh_basis
            )
    elif args.method == "MBIS":
        print("MBIS partitioning --")
        if args.lisa_basis is not None:
            raise ValueError("--lisa-basis applies only to LISA, not MBIS.")
        if args.proatom_basis is not None:
            raise ValueError("--proatom-basis applies only to HIRSHFELD, not MBIS.")
        if args.avh_basis is not None:
            raise ValueError("--avh-basis applies only to AVH, not MBIS.")
        pro_model_init = MBISProModel.from_geometry(data["atnums"], data["atcoords"], nshell_map)
    else:
        raise NotImplementedError
    cache = ComputeCache() if args.do_cache else None
    if args.method == "MBIS":
        pro_model, localgrids = optimize_reduce_pro_model(
            pro_model_init,
            grid,
            density,
            args.gtol,
            args.maxiter,
            args.density_cutoff,
            cache,
        )
    elif args.method == "LISA":
        pro_model, localgrids = optimize_pro_model(
            pro_model_init,
            grid,
            density,
            args.gtol,
            args.maxiter,
            args.density_cutoff,
            cache,
        )
    elif args.method == "AVH":
        pro_model, localgrids = optimize_avh_pro_model(
            pro_model_init,
            grid,
            density,
            args.gtol,
            args.maxiter,
            args.density_cutoff,
            cache,
        )
    elif args.method == "HIRSHFELD-I":
        optimizer = (
            optimize_spline_hirshfeld_i
            if isinstance(pro_model_init, SplineProModel)
            else optimize_hirshfeld_i
        )
        pro_model, localgrids = optimizer(
            pro_model_init,
            grid,
            density,
            args.gtol,
            args.maxiter,
            args.density_cutoff,
            cache,
        )
    else:
        pro_model = pro_model_init
        localgrids = [
            grid.get_localgrid(function.center, function.get_cutoff_radius(args.density_cutoff))
            for function in pro_model.fns
        ]
    print("Promodel")
    pro_model.pprint()
    print("Computing additional properties")
    results = pro_model.to_dict()
    radial_moments = compute_radial_moments(
        pro_model, grid, density, localgrids, args.density_cutoff, cache
    )
    charges = (
        np.asarray(data["atnums"], dtype=float) - radial_moments[:, 0]
        if args.method == "HIRSHFELD"
        else pro_model.charges
    )
    results.update(
        {
            "charges": charges,
            "radial_moments": radial_moments,
            "multipole_moments": compute_multipole_moments(
                pro_model, grid, density, localgrids, args.density_cutoff, cache
            ),
            "gtol": args.gtol,
            "maxiter": args.maxiter,
            "density_cutoff": args.density_cutoff,
        }
    )
    np.savez_compressed(args.out_npz, **results)
    print("Sum of charges: ", sum(charges))


def parse_nshell_arg(nshell):
    """Convert a list of nshell command-line arguments into a more convenient dictionary."""
    nshell_map = {}
    for word in nshell:
        if word.count(":") != 1:
            raise ValueError("Each nshell specification should have at least one colon.")
        atnum, num = word.split(":")
        nshell_map[int(atnum)] = int(num)
    return nshell_map


def parse_args(args=None):
    """Parse command-line arguments."""
    description = "Density partitioning of a given density on a grid."
    parser = argparse.ArgumentParser(prog="denspart", description=description)
    parser.add_argument("in_npz", help="The NPZ file with grid and density.")
    parser.add_argument("out_npz", help="The NPZ file in which resutls will be stored.")
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-8,
        help="Convergence tolerance for charge iteration or variational optimization. "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "-m",
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum number of charge or optimizer iterations. [default=%(default)s]",
    )
    parser.add_argument(
        "-c",
        "--density-cutoff",
        type=float,
        default=1e-10,
        help="Density cutoff, used to estimate local grid sizes. "
        "Set to zero for whole-grid integrations (molecules only). "
        "[default=%(default)s]",
    )
    parser.add_argument(
        "-t",
        "--method",
        type=str,
        choices=["HIRSHFELD", "HIRSHFELD-I", "AVH", "LISA", "MBIS"],
        default="MBIS",
        help="Partitioning method. [default=%(default)s]",
    )
    parser.add_argument(
        "--nshell",
        default=[],
        nargs="+",
        help="A whitespace-separate list of atnum:num items, e.g. 12:2 "
        "mean two shells for magnesium. "
        "The num part must be a positive integer and cannot exceed the "
        "default number of shells specified for that element. "
        "At least one argument must be given.",
    )
    parser.add_argument(
        "--lisa-basis",
        help="JSON file containing Gaussian basis functions for LISA. "
        "Both legacy HORTON-Part and aim-lisa-basis-v1 layouts are supported.",
    )
    parser.add_argument(
        "--proatom-basis",
        help="State-resolved Gaussian or aim-proatom-spline-v1 JSON file. "
        "HIRSHFELD selects the fixed neutral state; HIRSHFELD-I mixes adjacent states.",
    )
    parser.add_argument(
        "--avh-basis",
        help="Gaussian aim-avh-gaussian-v1 or radial-spline state library.",
    )
    parser.add_argument(
        "--avh-variant",
        choices=("A", "B", "M", "supplied", "a", "b", "m"),
        default="supplied",
        help="Select AVH-A/B/M states from a complete spline library, or use all supplied states.",
    )
    parser.add_argument(
        "--nocache",
        dest="do_cache",
        default=True,
        action="store_false",
        help="Disable caching. The cache increases memory consumption, "
        "it speeds up the calculation by about a factor of 2, "
        "and it could introduce more bugs.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main()
