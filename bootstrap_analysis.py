"""Command-line driver for the two-dimensional conformal bootstrap.

Examples
--------
The calibration point -- the bound here should come out near the exact 2d Ising
value ``Delta_epsilon = 1``::

    python bootstrap_analysis.py --delta-phi 0.125

Map the bound across a range and export it::

    python bootstrap_analysis.py --scan 0.0625,0.45,9 --csv bound.csv

Ask whether one specific assumption is excluded, and by how much::

    python bootstrap_analysis.py --delta-phi 0.125 --gap 1.4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

import bootstrap as bs

LOGGER = logging.getLogger("bootstrap_analysis")


def _build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exclusion bounds for 2d CFTs from crossing symmetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--delta-phi", type=float, default=0.125, help="external scalar dimension"
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="test one assumed scalar gap instead of computing the bound",
    )
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        metavar="LO,HI,N",
        help="map the bound across N values of delta-phi",
    )
    parser.add_argument("--max-order", type=int, default=9, help="highest derivative order")
    parser.add_argument("--max-spin", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=60, help="samples per spin")
    parser.add_argument("--audit-samples", type=int, default=400, help="samples in the audit grid")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="bisection tolerance")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json-summary", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stderr,
    )
    basis = bs.DerivativeBasis(max_order=args.max_order)
    common = dict(
        max_spin=args.max_spin,
        n_samples=args.n_samples,
        audit_samples=args.audit_samples,
    )
    LOGGER.info(
        "derivative basis: order <= %d, %d components %s",
        args.max_order,
        basis.size,
        basis.components[:4],
    )

    summary: dict[str, object] = {
        "max_order": args.max_order,
        "basis_size": basis.size,
        "max_spin": args.max_spin,
    }

    if args.gap is not None:
        started = time.perf_counter()
        result = bs.find_functional(args.delta_phi, args.gap, basis, **common)
        elapsed = time.perf_counter() - started
        print(f"\n=== delta_phi = {args.delta_phi}, assumed gap = {args.gap} ===")
        print(f"  excluded            : {result.found}")
        print(f"  optimal margin      : {result.margin:+.6e}")
        print(f"  constraints used    : {result.n_constraints}")
        print(f"  coefficient box hit : {result.box_active} "
              f"(if so, only the margin's sign is meaningful)")
        print(f"  elapsed             : {elapsed:.1f}s")
        if result.found:
            print("\n  This is a proof against the sampled spectrum: no unitary 2d CFT")
            print("  with this external dimension and this scalar gap can satisfy crossing.")
        else:
            print("\n  No functional found. That is NOT evidence such a CFT exists --")
            print("  only that this derivative basis cannot rule it out.")
        summary.update(
            {
                "delta_phi": args.delta_phi,
                "gap": args.gap,
                "excluded": result.found,
                "margin": result.margin,
                "n_constraints": result.n_constraints,
            }
        )
        _emit(summary, None, args)
        return 0

    if args.scan:
        try:
            low_text, high_text, count_text = args.scan.split(",")
            low, high, count = float(low_text), float(high_text), int(count_text)
        except ValueError as error:
            raise SystemExit(f"--scan expects 'LO,HI,N'; got {args.scan!r}") from error
        points = np.linspace(low, high, count)
    else:
        points = np.array([args.delta_phi])

    rows = []
    for delta_phi in points:
        started = time.perf_counter()
        bound = bs.scalar_gap_bound(
            float(delta_phi),
            basis,
            low=0.05,
            high=4.5,
            tolerance=args.tolerance,
            **common,
        )
        rows.append(
            {
                "delta_phi": float(delta_phi),
                "bound": bound,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        LOGGER.info("delta_phi = %.5f -> bound %.5f", delta_phi, bound)

    frame = pd.DataFrame(rows)
    print("\n=== upper bound on the leading scalar dimension (d = 2) ===")
    print(frame.to_string(index=False))

    ising = frame.loc[(frame["delta_phi"] - 0.125).abs() < 1e-9]
    if not ising.empty:
        value = float(ising["bound"].iloc[0])
        print(
            f"\n  2d Ising calibration: delta_sigma = 1/8 gives {value:.4f}; "
            f"the exact value is 1 (error {abs(value - 1):.4f})"
        )
        summary["ising_bound"] = value
        summary["ising_absolute_error"] = abs(value - 1.0)

    summary["points"] = rows
    _emit(summary, frame, args)
    return 0


def _emit(summary: dict[str, object], frame: pd.DataFrame | None, args) -> None:
    if args.csv and frame is not None:
        frame.to_csv(args.csv, index=False)
        LOGGER.info("wrote %s", args.csv)
    if args.json_summary:
        with open(args.json_summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        LOGGER.info("wrote %s", args.json_summary)


if __name__ == "__main__":
    raise SystemExit(main())
