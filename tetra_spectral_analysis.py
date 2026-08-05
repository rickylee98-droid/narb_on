#!/usr/bin/env python3
"""Spectral analysis of the graph Laplacian of a dense tetrahedral point cloud.

Builds a large cluster of unit-edge regular tetrahedra, merges coincident
vertices, forms the unit-distance graph, assembles the sparse combinatorial
Laplacian ``L = D - A``, and extracts the low end of its spectrum to hunt for
degeneracies and spectral gaps.

Three geometry backends are available, and the distinction matters:

``honeycomb`` (default)
    The tetrahedral-octahedral honeycomb on the FCC lattice.  Tetrahedra
    interlock face-to-face and genuinely *share* vertices and edges, so the
    merged unit-distance graph is connected and carries the full octahedral
    point group :math:`O_h`.  This is the configuration in which a spectral
    fingerprint exists: eigenvalue multiplicities are forced to match the
    dimensions of the irreducible representations of :math:`O_h` (1, 1, 2, 3,
    3).  Tetrahedra occupy exactly 1/3 of space here.

``ceg``
    The densest known packing of regular tetrahedra, ``phi = 4000/4671 ~
    0.856347``, taken as exact rationals from Chen, Engel & Glotzer (2010) and
    certified here by the separating-axis test rather than trusted.  Its
    spectral answer is complete and exact: the packing is built from *dimers*
    (two tetrahedra sharing a face), and that face contact is genuine vertex
    sharing, so each dimer contributes a 5-vertex component -- the triangular
    dipyramid, ``K5`` minus an edge -- while distinct dimers share nothing.
    The Laplacian spectrum of the whole packing is therefore exactly
    ``{0, 3, 5, 5, 5}`` repeated once per dimer.  The algebraic connectivity is
    exactly 0, every degeneracy is mere repetition of identical components, and
    there is no hidden symmetry group to find.  Maximising density and building
    a richly connected graph are simply different objectives.

``packing``
    A stochastic adaptive-shrinking-cell search, kept for exploring the
    density landscape; the achieved packing fraction is measured, never
    assumed, and reaches roughly 0.5-0.75 rather than the optimum.  With the
    default ``single`` motif the tetrahedra land in fully generic position and
    share no vertices at all, so the graph shatters further than the CEG case
    does -- into one disjoint ``K4`` per tetrahedron, spectrum ``{0, 4, 4, 4}``.

Examples
--------
Default run on the honeycomb::

    python tetra_spectral_analysis.py --min-tetrahedra 1000 --num-eigenvalues 100

Show the packing case, with a longer densification search::

    python tetra_spectral_analysis.py --backend packing --asc-cycles 3000

Export the spectrum for downstream work::

    python tetra_spectral_analysis.py --csv-prefix results/honeycomb --json-summary results/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import tetra_geometry as tg
import tetra_spectral as ts

LOGGER = logging.getLogger("tetra")

BANNER_WIDTH = 78


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def _rule(title: str = "", char: str = "=") -> str:
    if not title:
        return char * BANNER_WIDTH
    padded = f" {title} "
    side = max(BANNER_WIDTH - len(padded), 0)
    left = side // 2
    return char * left + padded + char * (side - left)


def _emit(text: str = "") -> None:
    print(text, flush=True)


def _emit_frame(frame: pd.DataFrame, floatfmt: str = "%.12g") -> None:
    if frame.empty:
        _emit("  (empty)")
        return
    _emit(frame.to_string(index=False, float_format=lambda v: floatfmt % v))


def configure_threads(threads: int | None) -> int:
    """Pin BLAS/LAPACK thread counts and report the effective value.

    The heavy numerical kernels here -- dense LAPACK on the Laplacian and the
    sparse factorisation inside shift-invert ARPACK -- are threaded inside
    BLAS, so this is where parallelism actually pays.  Thread-count environment
    variables are only read when the BLAS library loads, so ``threadpoolctl``
    is used when available to retune an already-imported library.
    """
    resolved = threads if threads and threads > 0 else (os.cpu_count() or 1)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(resolved)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=resolved)
        LOGGER.debug("threadpoolctl set BLAS limits to %d", resolved)
    except ImportError:
        LOGGER.debug("threadpoolctl unavailable; relying on environment variables")
    return resolved


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisResult:
    """Everything the pipeline produced, for reporting and export."""

    cloud: tg.TetraCloud
    merged: ts.MergedVertices
    bundle: ts.GraphBundle
    spectrum: ts.Spectrum
    degeneracy: ts.DegeneracyReport
    timings: dict[str, float]
    overlap_pairs: int | None


def build_cloud(args: argparse.Namespace) -> tg.TetraCloud:
    """Construct the tetrahedron cloud selected on the command line."""
    if args.backend == "honeycomb":
        return tg.build_honeycomb(args.min_tetrahedra)
    if args.backend == "ceg":
        return tg.build_ceg_packing(args.min_tetrahedra, variant=args.ceg_variant)
    if args.backend == "n3":
        return tg.build_n3_packing(args.min_tetrahedra)
    if args.backend == "p3":
        return tg.build_p3_packing(
            args.min_tetrahedra,
            screw=args.p3_screw,
            cycles=args.asc_cycles,
            restarts=args.asc_restarts,
            seed=args.seed,
        )
    return tg.build_dense_packing(
        args.min_tetrahedra,
        n_particles=args.asc_particles,
        motif=args.motif,
        cycles=args.asc_cycles,
        restarts=args.asc_restarts,
        seed=args.seed,
    )


def run_pipeline(args: argparse.Namespace) -> AnalysisResult:
    """Execute geometry -> graph -> Laplacian -> spectrum end to end."""
    timings: dict[str, float] = {}

    start = time.perf_counter()
    cloud = build_cloud(args)
    timings["geometry"] = time.perf_counter() - start
    cloud.assert_regular(edge=1.0, atol=1e-9)

    overlap_pairs: int | None = None
    if args.verify_packing:
        start = time.perf_counter()
        overlap_pairs = int(tg.find_overlapping_pairs(cloud.tetrahedra).shape[0])
        timings["overlap_verification"] = time.perf_counter() - start
        if overlap_pairs:
            LOGGER.error(
                "%d overlapping tetrahedron pairs detected -- the cloud is not a "
                "valid packing",
                overlap_pairs,
            )

    start = time.perf_counter()
    merged = ts.merge_vertices(cloud.raw_points, atol=args.merge_atol)
    timings["vertex_merge"] = time.perf_counter() - start

    start = time.perf_counter()
    bundle = ts.build_unit_distance_graph(
        merged.points, distance=1.0, atol=args.edge_atol
    )
    timings["graph_construction"] = time.perf_counter() - start

    start = time.perf_counter()
    laplacian = ts.graph_laplacian(bundle.adjacency)
    timings["laplacian"] = time.perf_counter() - start

    start = time.perf_counter()
    spectrum = ts.smallest_eigenvalues(
        laplacian,
        k=args.num_eigenvalues,
        dense_threshold=args.dense_threshold,
        tol=args.arpack_tol,
        maxiter=args.arpack_maxiter,
        n_components=bundle.n_components,
        component_labels=bundle.component_labels,
    )
    timings["eigensolve"] = time.perf_counter() - start

    start = time.perf_counter()
    degeneracy = ts.analyse_degeneracies(
        spectrum.eigenvalues,
        rtol=args.degeneracy_rtol,
        atol=args.degeneracy_atol,
    )
    timings["degeneracy_analysis"] = time.perf_counter() - start

    return AnalysisResult(
        cloud=cloud,
        merged=merged,
        bundle=bundle,
        spectrum=spectrum,
        degeneracy=degeneracy,
        timings=timings,
        overlap_pairs=overlap_pairs,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_geometry(result: AnalysisResult) -> None:
    cloud = result.cloud
    _emit(_rule("GEOMETRY"))
    rows = [
        ("tetrahedra", cloud.n_tetrahedra),
        ("raw vertices (4 per tetrahedron)", cloud.raw_points.shape[0]),
        ("edge-length max deviation", f"{cloud.max_edge_deviation():.3e}"),
        ("occupied volume", f"{cloud.occupied_volume:.6f}"),
    ]
    for key, value in cloud.provenance.items():
        rows.append((key.replace("_", " "), value))
    if result.overlap_pairs is not None:
        rows.append(("overlapping tetrahedron pairs", result.overlap_pairs))
    width = max(len(str(k)) for k, _ in rows)
    for key, value in rows:
        _emit(f"  {str(key):<{width}} : {value}")
    _emit()


def report_graph(result: AnalysisResult) -> None:
    merged, bundle = result.merged, result.bundle
    degrees = bundle.degrees
    _emit(_rule("VERTEX MERGE AND GRAPH"))
    rows = [
        ("raw vertices", merged.n_raw),
        ("distinct vertices after merge", merged.n_unique),
        ("compression ratio", f"{merged.n_raw / max(merged.n_unique, 1):.4f}"),
        ("max cluster radius", f"{merged.max_cluster_radius:.3e}"),
        ("largest merge multiplicity", int(merged.cluster_sizes.max())),
        ("graph nodes", bundle.n_nodes),
        ("graph edges", bundle.n_edges),
        ("connected components", bundle.n_components),
        ("edge density", f"{bundle.density:.6e}"),
        ("degree min / mean / max", f"{degrees.min()} / {degrees.mean():.4f} / {degrees.max()}"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        _emit(f"  {key:<{width}} : {value}")

    counts = np.bincount(degrees)
    hist = pd.DataFrame(
        {
            "degree": np.nonzero(counts)[0],
            "n_vertices": counts[counts > 0],
        }
    )
    hist["fraction"] = hist["n_vertices"] / bundle.n_nodes
    _emit()
    _emit("  Degree distribution:")
    _emit_frame(hist, floatfmt="%.6f")
    _emit()


def report_spectrum(result: AnalysisResult, top: int) -> None:
    spectrum, bundle = result.spectrum, result.bundle
    _emit(_rule("LAPLACIAN SPECTRUM"))

    # The kernel dimension equals the component count, but that can only be
    # *checked* when the computed window reaches past the kernel. If every
    # eigenvalue returned is zero, the kernel simply extends beyond the window
    # and there is nothing inconsistent about it.
    kernel_resolved = spectrum.k_returned > bundle.n_components
    if kernel_resolved:
        kernel_state = (
            "YES" if spectrum.zero_multiplicity == bundle.n_components else "NO -- MISMATCH"
        )
    elif spectrum.zero_multiplicity == spectrum.k_returned:
        kernel_state = f"not resolved (k={spectrum.k_returned} <= components)"
    else:
        kernel_state = "NO -- MISMATCH"
    kernel_ok = not kernel_state.startswith("NO")

    rows: list[tuple[str, Any]] = [
        ("Laplacian dimension", f"{spectrum.matrix_dimension} x {spectrum.matrix_dimension}"),
        ("eigenvalues requested", spectrum.k_requested),
        ("eigenvalues returned", spectrum.k_returned),
        ("solver path", spectrum.method),
        ("numerical zero tolerance", f"{spectrum.zero_tolerance:.3e}"),
        ("lambda_0", f"{spectrum.lambda_0:.15g}"),
        ("lambda_0 == 0 detected", "YES" if abs(spectrum.lambda_0) <= spectrum.zero_tolerance else "NO"),
        ("multiplicity of zero", spectrum.zero_multiplicity),
        ("connected components", bundle.n_components),
        ("kernel dimension matches components", kernel_state),
    ]
    if spectrum.k_returned >= 2:
        fiedler = spectrum.algebraic_connectivity
        rows.append(("lambda_1 (algebraic connectivity / Fiedler value)", f"{fiedler:.15g}"))
    first_positive = spectrum.first_positive
    rows.append(
        (
            "first strictly positive eigenvalue",
            "none in computed range" if first_positive is None else f"{first_positive:.15g}",
        )
    )
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        _emit(f"  {key:<{width}} : {value}")

    if not kernel_ok:
        _emit()
        _emit(
            "  WARNING: the number of zero eigenvalues disagrees with the component\n"
            "           count. Either the eigensolver lost accuracy or the zero\n"
            "           tolerance needs adjusting."
        )
    if bundle.n_components > 1:
        _emit()
        _emit(
            f"  NOTE: the graph has {bundle.n_components} components, so the Fiedler\n"
            "        value is exactly 0 by construction. The first strictly positive\n"
            "        eigenvalue above describes connectivity *within* a component."
        )
        if not kernel_resolved:
            _emit(
                f"        Only {spectrum.k_returned} eigenvalues were computed, all of\n"
                f"        them inside the {bundle.n_components}-dimensional kernel. Raise\n"
                f"        --num-eigenvalues above {bundle.n_components} to reach the\n"
                "        non-trivial part of the spectrum."
            )

    _emit()
    _emit(f"  First {top} eigenvalues with consecutive spacings:")
    _emit_frame(ts.spectrum_frame(spectrum, limit=top))
    _emit()


def report_degeneracy(result: AnalysisResult, top: int) -> None:
    report, spectrum = result.degeneracy, result.spectrum
    _emit(_rule("DEGENERACY STRUCTURE / ALGEBRAIC FINGERPRINT"))

    rows: list[tuple[str, Any]] = [
        ("eigenvalues analysed", spectrum.k_returned),
        ("distinct levels", report.n_levels),
        ("mean multiplicity", f"{spectrum.k_returned / max(report.n_levels, 1):.4f}"),
        ("max multiplicity", report.max_multiplicity),
        ("fraction of eigenvalues in degenerate levels", f"{report.degenerate_fraction:.4f}"),
    ]
    if report.gap_ratio_mean is not None:
        rows.append(("mean adjacent-gap ratio <r>", f"{report.gap_ratio_mean:.6f}"))
        rows.append(("  Poisson reference (uncorrelated)", f"{ts.POISSON_RATIO:.6f}"))
        rows.append(("  GOE reference (level repulsion)", f"{ts.GOE_RATIO:.6f}"))
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        _emit(f"  {key:<{width}} : {value}")

    _emit()
    _emit("  Multiplicity histogram:")
    _emit_frame(ts.multiplicity_frame(report), floatfmt="%.6f")

    _emit()
    _emit(f"  First {top} distinct levels:")
    _emit_frame(ts.level_frame(report, limit=top))

    if report.raw_spacings.size:
        gaps = report.raw_spacings
        nonzero = gaps[gaps > spectrum.zero_tolerance]
        _emit()
        _emit("  Consecutive spacing statistics (Delta lambda_n):")
        stats = pd.DataFrame(
            {
                "statistic": ["count", "min", "median", "mean", "max", "std"],
                "all_spacings": [
                    gaps.size,
                    gaps.min(),
                    float(np.median(gaps)),
                    gaps.mean(),
                    gaps.max(),
                    gaps.std(ddof=0),
                ],
                "nonzero_spacings": [
                    nonzero.size,
                    nonzero.min() if nonzero.size else np.nan,
                    float(np.median(nonzero)) if nonzero.size else np.nan,
                    nonzero.mean() if nonzero.size else np.nan,
                    nonzero.max() if nonzero.size else np.nan,
                    nonzero.std(ddof=0) if nonzero.size else np.nan,
                ],
            }
        )
        _emit_frame(stats)
    _emit()


def report_interpretation(result: AnalysisResult) -> None:
    """State what the numbers do and do not support."""
    report, bundle, spectrum = result.degeneracy, result.bundle, result.spectrum
    _emit(_rule("INTERPRETATION"))

    if bundle.n_components > 1 and report.max_multiplicity > 1:
        component_sizes = np.bincount(bundle.component_labels)
        uniform = component_sizes.min() == component_sizes.max()
        _emit(
            f"  The graph is disconnected ({bundle.n_components} components"
            + (f", all of size {component_sizes.min()}" if uniform else "")
            + ").\n"
            "  Degeneracy here is trivial: identical components contribute identical\n"
            "  eigenvalues. This is repetition, not a hidden symmetry group."
        )
    elif report.max_multiplicity > 1:
        oh_like = sum(
            count for mult, count in report.multiplicity_histogram.items() if mult in (1, 2, 3)
        )
        share = oh_like / max(report.n_levels, 1)
        _emit(
            "  The graph is connected, so degeneracies reflect the point symmetry of\n"
            "  the cluster rather than repeated disjoint pieces.\n"
            f"  {share:.1%} of distinct levels have multiplicity in {{1, 2, 3}}, the\n"
            "  irreducible representation dimensions of the octahedral group O_h.\n"
            "  That is the expected fingerprint for an FCC-derived cluster: the\n"
            "  symmetry is the finite point group O_h (order 48), not a continuous\n"
            "  Lie group. Multiplicities above 3 indicate accidental coincidences\n"
            "  between distinct irreps at the resolution of the degeneracy tolerance."
        )
    else:
        _emit(
            "  No degenerate levels were resolved at the current tolerance, so the\n"
            "  spectrum shows no multiplicity structure to attribute to a symmetry\n"
            "  group. Try loosening --degeneracy-rtol, or use a cluster whose\n"
            "  boundary respects the lattice point group."
        )

    if report.gap_ratio_mean is not None:
        _emit()
        _emit(
            f"  The mean adjacent-gap ratio <r> = {report.gap_ratio_mean:.4f} sits"
            + (
                " near the Poisson value,\n  which is what superposed independent symmetry sectors produce."
                if abs(report.gap_ratio_mean - ts.POISSON_RATIO)
                <= abs(report.gap_ratio_mean - ts.GOE_RATIO)
                else " nearer the GOE value,\n  indicating level repulsion rather than clean sector decomposition."
            )
        )

    gap = spectrum.first_positive
    if gap is not None and bundle.n_components == 1:
        _emit()
        _emit(
            f"  The spectral gap is {gap:.6g}. For a finite cluster cut from an\n"
            "  infinite lattice this scales as O(1/L^2) with cluster diameter L --\n"
            "  it is a finite-size effect, not evidence of an intrinsic mass gap."
        )
    _emit()


def report_timings(result: AnalysisResult, threads: int) -> None:
    _emit(_rule("EXECUTION LOG"))
    total = sum(result.timings.values())
    frame = pd.DataFrame(
        {
            "stage": list(result.timings.keys()),
            "seconds": list(result.timings.values()),
        }
    )
    frame["percent"] = 100.0 * frame["seconds"] / max(total, 1e-12)
    _emit_frame(frame, floatfmt="%.4f")
    _emit()
    _emit(f"  total: {total:.4f} s   |   BLAS threads: {threads}")
    _emit(
        f"  python {platform.python_version()} | numpy {np.__version__} | "
        f"pandas {pd.__version__}"
    )
    _emit()


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_csv(result: AnalysisResult, prefix: str) -> list[Path]:
    base = Path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    targets = {
        f"{base.name}_eigenvalues.csv": ts.spectrum_frame(result.spectrum),
        f"{base.name}_levels.csv": ts.level_frame(result.degeneracy),
        f"{base.name}_multiplicities.csv": ts.multiplicity_frame(result.degeneracy),
    }
    for name, frame in targets.items():
        path = base.parent / name
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def summary_dict(result: AnalysisResult, args: argparse.Namespace) -> dict[str, Any]:
    spectrum, bundle, report = result.spectrum, result.bundle, result.degeneracy
    return {
        "parameters": {
            "backend": args.backend,
            "min_tetrahedra": args.min_tetrahedra,
            "num_eigenvalues": args.num_eigenvalues,
            "merge_atol": args.merge_atol,
            "edge_atol": args.edge_atol,
            "degeneracy_rtol": args.degeneracy_rtol,
            "degeneracy_atol": args.degeneracy_atol,
            "seed": args.seed,
            "motif": args.motif,
            "ceg_variant": args.ceg_variant,
        },
        "geometry": {
            "n_tetrahedra": result.cloud.n_tetrahedra,
            "provenance": {k: v for k, v in result.cloud.provenance.items()},
            "overlapping_pairs": result.overlap_pairs,
        },
        "graph": {
            "n_raw_vertices": result.merged.n_raw,
            "n_distinct_vertices": result.merged.n_unique,
            "max_merge_cluster_radius": result.merged.max_cluster_radius,
            "n_nodes": bundle.n_nodes,
            "n_edges": bundle.n_edges,
            "n_components": bundle.n_components,
            "degree_min": int(bundle.degrees.min()),
            "degree_max": int(bundle.degrees.max()),
            "degree_mean": float(bundle.degrees.mean()),
        },
        "spectrum": {
            "matrix_dimension": spectrum.matrix_dimension,
            "method": spectrum.method,
            "k_returned": spectrum.k_returned,
            "zero_tolerance": spectrum.zero_tolerance,
            "lambda_0": spectrum.lambda_0,
            "zero_multiplicity": spectrum.zero_multiplicity,
            "algebraic_connectivity": (
                spectrum.algebraic_connectivity if spectrum.k_returned >= 2 else None
            ),
            "first_positive_eigenvalue": spectrum.first_positive,
            "first_20_eigenvalues": spectrum.eigenvalues[:20].tolist(),
        },
        "degeneracy": {
            "n_levels": report.n_levels,
            "max_multiplicity": report.max_multiplicity,
            "degenerate_fraction": report.degenerate_fraction,
            "multiplicity_histogram": report.multiplicity_histogram,
            "gap_ratio_mean": report.gap_ratio_mean,
            "poisson_reference": ts.POISSON_RATIO,
            "goe_reference": ts.GOE_RATIO,
        },
        "timings_seconds": result.timings,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tetra_spectral_analysis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    geom = parser.add_argument_group("geometry")
    geom.add_argument(
        "--backend",
        choices=("honeycomb", "ceg", "n3", "p3", "packing"),
        default="honeycomb",
        help=(
            "cluster construction: 'honeycomb' interlocking FCC honeycomb "
            "(default), 'ceg' the exact Chen-Engel-Glotzer optimum at phi = "
            "4000/4671, 'n3' the N = 3 phase at exactly 2/3, 'p3' a three-fold "
            "screw-symmetric search for the N = 3 "
            "phase, 'packing' a general stochastic adaptive-shrinking-cell search"
        ),
    )
    geom.add_argument(
        "--min-tetrahedra",
        type=int,
        default=1000,
        help="minimum number of tetrahedra in the cloud (default: 1000)",
    )
    geom.add_argument(
        "--asc-cycles",
        type=int,
        default=1500,
        help="Monte Carlo cycles for the packing backend (default: 1500)",
    )
    geom.add_argument(
        "--ceg-variant",
        choices=tuple(tg.CEG_FAMILY_PRESETS),
        default="optimal",
        help=(
            "which member of the Chen-Engel-Glotzer double dimer family to build: "
            "'optimal' phi=4000/4671 (default), 'optimal-mirror' the same density "
            "via the mirror point, 'kallus-elser-gravel' phi=100/117, "
            "'torquato-jiao' phi=12250/14319"
        ),
    )
    geom.add_argument(
        "--p3-screw",
        type=int,
        choices=tg.P3_SCREW_INDICES,
        default=None,
        help=(
            "screw index for the p3 backend: 0 = P3, 1 = P3_1, 2 = P3_2 "
            "(default: search all three and keep the densest)"
        ),
    )
    geom.add_argument(
        "--motif",
        choices=tg.MOTIF_NAMES,
        default="dimer",
        help=(
            "rigid body the packing search moves: 'dimer' packs face-fused "
            "triangular dipyramids (the motif of every known optimum), 'single' "
            "packs free tetrahedra (default: dimer)"
        ),
    )
    geom.add_argument(
        "--asc-restarts",
        type=int,
        default=4,
        help="independent packing searches; the densest is kept (default: 4)",
    )
    geom.add_argument(
        "--asc-particles",
        type=int,
        default=2,
        help=(
            "rigid motifs per periodic cell for the packing backend; with the "
            "default dimer motif this is 2 motifs = 4 tetrahedra (default: 2)"
        ),
    )
    geom.add_argument(
        "--seed", type=int, default=20250805, help="random seed (default: 20250805)"
    )
    geom.add_argument(
        "--verify-packing",
        action="store_true",
        help="run an exact separating-axis check that no tetrahedra interpenetrate",
    )

    graph = parser.add_argument_group("graph")
    graph.add_argument(
        "--merge-atol",
        type=float,
        default=1e-5,
        help="Euclidean tolerance for merging coincident vertices (default: 1e-5)",
    )
    graph.add_argument(
        "--edge-atol",
        type=float,
        default=1e-6,
        help="tolerance on unit separation when creating edges (default: 1e-6)",
    )

    spec = parser.add_argument_group("spectrum")
    spec.add_argument(
        "--num-eigenvalues",
        type=int,
        default=100,
        help="number of smallest eigenvalues to compute (default: 100)",
    )
    spec.add_argument(
        "--dense-threshold",
        type=int,
        default=2000,
        help="use dense LAPACK at or below this matrix dimension (default: 2000)",
    )
    spec.add_argument(
        "--arpack-tol",
        type=float,
        default=0.0,
        help="ARPACK relative tolerance; 0 means machine precision (default: 0)",
    )
    spec.add_argument(
        "--arpack-maxiter",
        type=int,
        default=None,
        help="ARPACK iteration cap (default: 10 * matrix dimension)",
    )
    spec.add_argument(
        "--degeneracy-rtol",
        type=float,
        default=1e-8,
        help="relative tolerance grouping eigenvalues into levels (default: 1e-8)",
    )
    spec.add_argument(
        "--degeneracy-atol",
        type=float,
        default=1e-9,
        help="absolute tolerance grouping eigenvalues into levels (default: 1e-9)",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--top", type=int, default=20, help="rows shown in the head tables (default: 20)"
    )
    out.add_argument("--csv-prefix", type=str, default=None, help="write CSV tables with this prefix")
    out.add_argument("--json-summary", type=str, default=None, help="write a JSON summary here")
    out.add_argument("--threads", type=int, default=None, help="BLAS thread count (default: all cores)")
    out.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    return parser


def validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject argument combinations that cannot produce a meaningful result."""
    if args.min_tetrahedra < 1:
        parser.error("--min-tetrahedra must be >= 1")
    if args.num_eigenvalues < 1:
        parser.error("--num-eigenvalues must be >= 1")
    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.merge_atol <= 0.0:
        parser.error("--merge-atol must be positive")
    if args.edge_atol < 0.0:
        parser.error("--edge-atol must be non-negative")
    if args.edge_atol >= 1.0:
        parser.error("--edge-atol must be smaller than the unit edge length")
    if args.merge_atol >= 0.5:
        parser.error("--merge-atol must be far below the unit edge length")
    if args.asc_cycles < 1:
        parser.error("--asc-cycles must be >= 1")
    if args.asc_particles < 1:
        parser.error("--asc-particles must be >= 1")
    if args.asc_restarts < 1:
        parser.error("--asc-restarts must be >= 1")
    if args.dense_threshold < 0:
        parser.error("--dense-threshold must be non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    threads = configure_threads(args.threads)

    _emit(_rule("TETRAHEDRON PACKING :: GRAPH LAPLACIAN SPECTRAL ANALYSIS"))
    _emit(f"  backend={args.backend}  min_tetrahedra={args.min_tetrahedra}  "
          f"k={args.num_eigenvalues}  seed={args.seed}")
    _emit()

    try:
        result = run_pipeline(args)
    except (ValueError, TypeError, RuntimeError, AssertionError, MemoryError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1

    report_geometry(result)
    report_graph(result)
    report_spectrum(result, args.top)
    report_degeneracy(result, args.top)
    report_interpretation(result)
    report_timings(result, threads)

    if args.csv_prefix:
        for path in export_csv(result, args.csv_prefix):
            _emit(f"  wrote {path}")
    if args.json_summary:
        path = Path(args.json_summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary_dict(result, args), indent=2, default=str))
        _emit(f"  wrote {path}")

    if result.overlap_pairs:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
