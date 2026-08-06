"""Command-line driver for exact ``k = 1`` amplituhedron tiling analysis.

Enumerates every tiling of ``A(n, 1, m) = C(n, m)``, verifies each one exactly,
builds the bistellar flip graph, and reports its Laplacian spectrum.

Examples
--------
Reproduce the Catalan numbers, which is the pipeline's calibration case::

    python amplituhedron_analysis.py --n 8 --m 2

The physical case, with the exact spectrum of the flip graph::

    python amplituhedron_analysis.py --n 8 --m 4 --exact-spectrum

The hypersimplex, whose volume must come out as an Eulerian number::

    python amplituhedron_analysis.py --hypersimplex 2,5 --volume-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import amplituhedron as amp
import tetra_spectral as ts

LOGGER = logging.getLogger("amplituhedron_analysis")


def _build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact tilings of the k = 1 amplituhedron and their flip-graph spectrum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=7, help="number of vertices")
    parser.add_argument("--m", type=int, default=4, help="dimension (4 is physical, 2 is the toy case)")
    parser.add_argument(
        "--hypersimplex",
        type=str,
        default=None,
        metavar="K,N",
        help="analyse Delta(k, n) instead of the cyclic polytope",
    )
    parser.add_argument(
        "--volume-only",
        action="store_true",
        help="compute the normalised volume and stop, skipping enumeration",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many tilings"
    )
    parser.add_argument(
        "--exact-spectrum",
        action="store_true",
        help="diagonalise the Laplacian symbolically, so multiplicities carry no tolerance",
    )
    parser.add_argument(
        "--automorphisms",
        action="store_true",
        help="count flip-graph automorphisms (expensive beyond a few hundred tilings)",
    )
    parser.add_argument("--top", type=int, default=20, help="eigenvalues to print")
    parser.add_argument("--csv-prefix", type=str, default=None, help="write frames to CSV")
    parser.add_argument("--json-summary", type=str, default=None, help="write the summary as JSON")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def _configuration(args: argparse.Namespace) -> tuple[amp.IntMatrix, str, int | None]:
    """Build the point configuration, with an independent volume where one exists."""
    if args.hypersimplex is not None:
        try:
            k_text, n_text = args.hypersimplex.split(",")
            k, n = int(k_text), int(n_text)
        except ValueError as error:
            raise SystemExit(
                f"--hypersimplex expects 'k,n'; got {args.hypersimplex!r}"
            ) from error
        vertices = amp.hypersimplex(k, n)
        LOGGER.info(
            "hypersimplex Delta(%d, %d): %d vertices in dimension %d",
            k, n, len(vertices), len(vertices[0]),
        )
        return vertices, f"Delta({k},{n})", None

    vertices = amp.cyclic_polytope(args.n, args.m)
    facets = amp.gale_facets(args.n, args.m)
    volume = amp.polytope_normalised_volume(vertices, facets)
    LOGGER.info(
        "A(%d, 1, %d) = C(%d, %d): %d facets by Gale's condition, normalised volume %d",
        args.n, args.m, args.n, args.m, len(facets), volume,
    )
    return vertices, f"A({args.n},1,{args.m})", volume


def _spectrum_frame(graph: nx.Graph, exact: bool) -> tuple[pd.DataFrame, dict[str, object]]:
    """Laplacian spectrum of the flip graph, numerically or symbolically."""
    laplacian = nx.laplacian_matrix(graph).todense().astype(int)
    if exact:
        import sympy

        eigenvalues = sympy.Matrix(laplacian.tolist()).eigenvals()
        rows = []
        for value, multiplicity in eigenvalues.items():
            rows.append(
                {
                    "eigenvalue": float(value),
                    # sympy's eigenvalues are already exact algebraic numbers.
                    # Passing them through nsimplify would *lose* exactness: it
                    # fits a closed form to a float and happily returns things
                    # like 2**(167/459)*3**(146/153) for an algebraic irrational.
                    "exact": str(value),
                    "multiplicity": int(multiplicity),
                    "is_integer": bool(value.is_integer),
                }
            )
        frame = pd.DataFrame(rows).sort_values("eigenvalue").reset_index(drop=True)
        largest = frame.loc[frame["multiplicity"].idxmax()]
        summary = {
            "spectrum_method": "exact",
            "n_distinct_levels": int(len(frame)),
            "max_multiplicity": int(frame["multiplicity"].max()),
            "max_multiplicity_eigenvalue": str(largest["exact"]),
            "integer_eigenvalues": int(frame["is_integer"].sum()),
        }
        return frame, summary

    values = np.linalg.eigvalsh(np.asarray(laplacian, dtype=float))
    report = ts.analyse_degeneracies(values)
    frame = pd.DataFrame(
        {"index": np.arange(values.size), "eigenvalue": values}
    )
    summary = {
        "spectrum_method": "numeric",
        "n_distinct_levels": int(report.n_levels),
        "multiplicity_histogram": {int(k): int(v) for k, v in report.multiplicity_histogram.items()},
        "max_multiplicity": int(report.max_multiplicity),
        "gap_ratio_mean": float(report.gap_ratio_mean)
        if report.gap_ratio_mean is not None
        else None,
    }
    return frame, summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stderr,
    )

    try:
        vertices, label, known_volume = _configuration(args)
    except ValueError as error:
        LOGGER.error("could not build the configuration: %s", error)
        return 2

    started = time.perf_counter()
    volume = known_volume if known_volume is not None else amp.configuration_volume(vertices)
    LOGGER.info("normalised volume %d (%.2fs)", volume, time.perf_counter() - started)

    summary: dict[str, object] = {
        "configuration": label,
        "n_vertices": len(vertices),
        "dimension": len(vertices[0]),
        "normalised_volume": volume,
    }

    if args.hypersimplex is not None:
        k, n = (int(part) for part in args.hypersimplex.split(","))
        expected = amp.eulerian_number(n - 1, k - 1)
        summary["eulerian_number"] = expected
        summary["matches_eulerian_number"] = volume == expected
        print(f"{label}: normalised volume {volume}, Eulerian A({n-1},{k-1}) = {expected}")
        if volume != expected:
            LOGGER.error("volume disagrees with the Eulerian number -- this is a bug")
            return 1

    if args.volume_only:
        _emit(summary, None, None, args)
        return 0

    started = time.perf_counter()
    report = amp.enumerate_tilings(vertices, volume, limit=args.limit)
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "%d tilings from %d candidate simplices (%.2fs)",
        report.n_tilings, len(report.candidates), elapsed,
    )

    for tiling in report.tilings:
        if not amp.is_tiling(vertices, tiling, volume):
            LOGGER.error("an enumerated family failed verification -- this is a bug")
            return 1

    graph = amp.flip_graph(vertices, report.tilings)
    degrees = [d for _, d in graph.degree()]
    summary.update(
        {
            "n_candidate_simplices": len(report.candidates),
            "n_tilings": report.n_tilings,
            "tiling_sizes": report.sizes,
            "all_verified": True,
            "enumeration_seconds": round(elapsed, 3),
            "flip_graph_edges": graph.number_of_edges(),
            "flip_graph_connected": bool(nx.is_connected(graph)) if graph.number_of_nodes() else False,
            "flip_degree_min": min(degrees) if degrees else 0,
            "flip_degree_max": max(degrees) if degrees else 0,
        }
    )

    if args.hypersimplex is None and args.m == 2:
        expected = amp.catalan(args.n - 2)
        summary["catalan_expected"] = expected
        summary["matches_catalan"] = report.n_tilings == expected and args.limit is None

    if args.automorphisms:
        from networkx.algorithms.isomorphism import GraphMatcher

        order = sum(1 for _ in GraphMatcher(graph, graph).isomorphisms_iter())
        summary["flip_graph_automorphisms"] = order
        summary["matches_dihedral_2n"] = order == 2 * len(vertices)

    spectrum, spectral_summary = _spectrum_frame(graph, args.exact_spectrum)
    summary.update(spectral_summary)

    print(f"\n=== {label} ===")
    print(f"  candidate simplices : {len(report.candidates)}")
    print(f"  normalised volume   : {volume}")
    print(f"  tilings             : {report.n_tilings}  (sizes {report.sizes})")
    print(f"  every tiling verified exactly: {summary['all_verified']}")
    print(f"  flip graph          : {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, connected={summary['flip_graph_connected']}")
    print(f"  flip degrees        : {summary['flip_degree_min']}..{summary['flip_degree_max']}")
    if "matches_catalan" in summary:
        print(f"  Catalan check       : {summary['catalan_expected']} -> {summary['matches_catalan']}")
    if "flip_graph_automorphisms" in summary:
        print(f"  |Aut(flip graph)|   : {summary['flip_graph_automorphisms']} "
              f"(dihedral 2n = {2 * len(vertices)}: {summary['matches_dihedral_2n']})")
    print("\n  spectrum:")
    print(spectrum.head(args.top).to_string(index=False))

    _emit(summary, spectrum, report, args)
    return 0


def _emit(
    summary: dict[str, object],
    spectrum: pd.DataFrame | None,
    report: amp.TilingReport | None,
    args: argparse.Namespace,
) -> None:
    if args.csv_prefix and spectrum is not None:
        spectrum.to_csv(f"{args.csv_prefix}_spectrum.csv", index=False)
        if report is not None:
            pd.DataFrame(
                {
                    "tiling": [" ".join(map(str, sorted(t))) for t in report.tilings],
                    "n_simplices": [len(t) for t in report.tilings],
                }
            ).to_csv(f"{args.csv_prefix}_tilings.csv", index=False)
        LOGGER.info("wrote CSV frames with prefix %s", args.csv_prefix)
    if args.json_summary:
        with open(args.json_summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        LOGGER.info("wrote %s", args.json_summary)


if __name__ == "__main__":
    raise SystemExit(main())
