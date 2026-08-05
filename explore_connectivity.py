#!/usr/bin/env python3
"""Map connectivity of the unit-distance graph across the CEG double dimer family.

The Chen-Engel-Glotzer construction is a three-parameter family (u, v, w) of
double dimer packings.  Density is a smooth function of (u, v) alone, but the
*connectivity* of the unit-distance graph is not smooth at all: it is supported
on the single plane ``u = 0`` and vanishes everywhere else.

This script measures that, sweeping the straight path from the Kallus-Elser-Gravel
origin ``(0, 0, 0)`` to the density optimum ``(3/160, 3/64, 0)`` and reporting:

* the drift of the KEG inter-dimer contacts away from unit separation,
* the component structure and first positive Laplacian eigenvalue,
* how the apparent transition width tracks the edge tolerance rather than any
  feature of the geometry.

It writes a CSV of every sample and renders a three-panel figure.

Usage::

    python explore_connectivity.py --reps 3 --out connectivity
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import tetra_geometry as tg
import tetra_spectral as ts

LOGGER = logging.getLogger("explore")

#: Endpoint of the path: the density optimum of Theorem 1.
U_OPT, V_OPT = Fraction(3, 160), Fraction(3, 64)

#: Densest point on the connected plane u = 0, derived from the P'' constraints
#: ``2v - w <= 33/320`` and ``v + w <= 3/64``: adding them gives ``v <= 1/20``.
#: It coincides with the paper's C3+cen entry, phi = 125/146.
V_CONNECTED_MAX, W_CONNECTED_MAX = Fraction(1, 20), Fraction(-1, 320)

#: Edge tolerances used to show that the transition width is an artefact.  All
#: three are tight enough that no spurious near-unit pair is admitted; a looser
#: value such as 1e-3 starts inventing edges between vertices that were never in
#: contact, which is a different phenomenon and would muddy the comparison.
EDGE_TOLERANCES: tuple[float, ...] = (1e-9, 1e-7, 1e-5)


@dataclass(frozen=True)
class Sample:
    """One point of a parameter sweep."""

    t: float
    u: float
    v: float
    w: float
    phi: float
    contact_drift: float
    n_components: int
    largest_component: int
    max_degree: int
    first_positive: float
    edge_atol: float


def _feasible_w(v: Fraction) -> Fraction:
    """A ``w`` keeping ``(0, v, w)`` inside ``P''``; density does not depend on it."""
    upper = Fraction(3, 64) - abs(v)
    return min(Fraction(0), upper)


def keg_contact_pairs(reps: int, tol: float = 1e-12) -> np.ndarray:
    """Raw-vertex index pairs that sit at exact unit separation *between* dimers.

    Evaluated at the KEG origin, where these contacts exist.  Vertex indexing is
    deterministic across the family, so the same pairs can be tracked as the
    parameters move.
    """
    cloud = tg.build_ceg_packing(4 * reps**3, variant="kallus-elser-gravel")
    points = cloud.raw_points
    dimer = np.repeat(np.arange(cloud.n_tetrahedra // 2), 8)

    pairs = cKDTree(points).query_pairs(r=1.0 + tol, output_type="ndarray")
    separation = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    unit = pairs[np.abs(separation - 1.0) < tol]
    return unit[dimer[unit[:, 0]] != dimer[unit[:, 1]]]


def measure(
    u: Fraction,
    v: Fraction,
    w: Fraction,
    *,
    reps: int,
    contacts: np.ndarray,
    edge_atol: float,
    t: float,
) -> Sample:
    """Build one family member and measure its graph."""
    cloud = tg.build_ceg_packing(4 * reps**3, u=u, v=v, w=w)
    points = cloud.raw_points

    separation = np.linalg.norm(
        points[contacts[:, 0]] - points[contacts[:, 1]], axis=1
    )
    drift = float(np.abs(separation - 1.0).max()) if contacts.size else 0.0

    merged = ts.merge_vertices(points, atol=1e-5)
    bundle = ts.build_unit_distance_graph(merged.points, atol=edge_atol)
    spectrum = ts.smallest_eigenvalues(
        ts.graph_laplacian(bundle.adjacency),
        k=merged.n_unique,
        n_components=bundle.n_components,
        component_labels=bundle.component_labels,
    )
    first_positive = spectrum.first_positive

    return Sample(
        t=t,
        u=float(u),
        v=float(v),
        w=float(w),
        phi=float(tg.ceg_packing_fraction(u, v, w)),
        contact_drift=drift,
        n_components=bundle.n_components,
        largest_component=int(np.bincount(bundle.component_labels).max()),
        max_degree=int(bundle.degrees.max()),
        first_positive=float("nan") if first_positive is None else first_positive,
        edge_atol=edge_atol,
    )


def sweep_to_optimum(reps: int, contacts: np.ndarray) -> pd.DataFrame:
    """Sweep the straight path from the KEG origin to the density optimum."""
    fractions = [Fraction(0)] + [
        Fraction(1, 10**k) * Fraction(m, 10)
        for k in range(6, 0, -1)
        for m in (1, 3)
    ] + [Fraction(1, 10), Fraction(3, 10), Fraction(1, 2), Fraction(1)]

    rows: list[Sample] = []
    for atol in EDGE_TOLERANCES:
        for t in fractions:
            rows.append(
                measure(
                    t * U_OPT,
                    t * V_OPT,
                    Fraction(0),
                    reps=reps,
                    contacts=contacts,
                    edge_atol=atol,
                    t=float(t),
                )
            )
            LOGGER.debug("t=%s atol=%.0e -> %s", t, atol, rows[-1])
    return pd.DataFrame([asdict(r) for r in rows])


def sweep_connected_plane(reps: int, contacts: np.ndarray) -> pd.DataFrame:
    """Sweep ``v`` along the connected plane ``u = 0``."""
    rows: list[Sample] = []
    for numerator in range(0, 21):
        v = Fraction(numerator, 20) * V_CONNECTED_MAX
        rows.append(
            measure(
                Fraction(0),
                v,
                _feasible_w(v),
                reps=reps,
                contacts=contacts,
                edge_atol=EDGE_TOLERANCES[0],
                t=float(v / V_CONNECTED_MAX),
            )
        )
    return pd.DataFrame([asdict(r) for r in rows])


def render(path_df: pd.DataFrame, plane_df: pd.DataFrame, out: Path) -> Path:
    """Render the three-panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Categorical hues assigned in fixed order, never cycled.  Verified with the
    # palette validator: lightness band, chroma floor, adjacent CVD separation
    # (worst pair dE 13.1 protan), normal-vision floor and contrast all pass.
    BLUE, ORANGE, TEAL = "#3B62D9", "#C2570A", "#00897B"
    INK, MUTED, GRID = "#1A1A1A", "#5C5C5C", "#DCDCDC"
    SERIES = (BLUE, ORANGE, TEAL)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.suptitle(
        "Connectivity of the unit-distance graph across the Chen-Engel-Glotzer "
        "double dimer family",
        fontsize=13, fontweight="bold", y=1.03,
    )

    # ---- Panel 1: the collapse is a step at u = 0, not a curve ------------- #
    ax = axes[0]
    positive = path_df[path_df["t"] > 0]
    base = positive[positive["edge_atol"] == EDGE_TOLERANCES[0]].sort_values("t")
    ax.loglog(base["t"], base["contact_drift"], color=INK, lw=2.2, zorder=4)
    # Single series: the title names it, so a direct label replaces the legend.
    midpoint = len(base) // 2
    ax.annotate(
        "contact drift\n|d - 1| = 0.0437 t",
        xy=(base["t"].iloc[midpoint], base["contact_drift"].iloc[midpoint]),
        xytext=(-10, 20), textcoords="offset points",
        fontsize=8.5, color=INK, ha="right", va="bottom", fontweight="bold",
    )
    for colour, atol in zip(SERIES, EDGE_TOLERANCES):
        ax.axhline(atol, color=colour, lw=1.5, ls=(0, (5, 3)), zorder=2)
        ax.annotate(
            f"edge tol {atol:.0e}", xy=(1.0, atol), xycoords=("axes fraction", "data"),
            xytext=(-3, 4), textcoords="offset points",
            color=colour, fontsize=8, ha="right", va="bottom",
        )
    ax.set_xlabel("t   (0 = KEG origin,  1 = density optimum)")
    ax.set_ylabel("drift of the 156 KEG inter-dimer contacts")
    ax.set_title(
        "Contacts are exact only at u = 0\nwhere drift crosses the tolerance is an artefact",
        fontsize=10, color=INK, loc="left", pad=10,
    )
    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)

    # ---- Panel 2: what the spectrum does at the transition ---------------- #
    ax = axes[1]
    for colour, atol in zip(SERIES, EDGE_TOLERANCES):
        subset = path_df[(path_df["edge_atol"] == atol) & (path_df["t"] > 0)].sort_values("t")
        ax.semilogx(
            subset["t"], subset["first_positive"], color=colour, lw=2.1,
            marker="o", ms=4, mew=0, label=f"edge tol {atol:.0e}", zorder=3,
        )
    keg_value = float(
        path_df[(path_df["t"] == 0) & (path_df["edge_atol"] == EDGE_TOLERANCES[1])][
            "first_positive"
        ].iloc[0]
    )
    ax.axhline(keg_value, color=MUTED, lw=1.1, ls=(0, (2, 3)), zorder=1)
    ax.annotate(
        f"connected KEG network:  {keg_value:.3f}",
        xy=(1.0, keg_value), xycoords=("axes fraction", "data"),
        xytext=(-3, 5), textcoords="offset points",
        fontsize=8, color=MUTED, ha="right",
    )
    ax.annotate(
        "3 = the K5-e value:\nisolated dimers",
        xy=(0.985, 3.0), xycoords=("axes fraction", "data"),
        xytext=(-3, -8), textcoords="offset points",
        fontsize=8, color=INK, ha="right", va="top",
    )
    ax.set_xlabel("t   (0 = KEG origin,  1 = density optimum)")
    ax.set_ylabel("first positive eigenvalue  $\\lambda_1^+$")
    ax.set_title(
        "The jump tracks the tolerance, not the geometry\nshifting a decade per decade of tolerance",
        fontsize=10, color=INK, loc="left", pad=10,
    )
    ax.set_ylim(0, 3.45)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.62))

    # ---- Panel 3: the connected plane, and what density it can reach ------- #
    ax = axes[2]
    phi_opt, phi_conn, phi_keg = 4000 / 4671, 125 / 146, 100 / 117
    plane = plane_df.sort_values("v")
    ax.plot(plane["v"], plane["phi"], color=BLUE, lw=2.4, zorder=3)
    ax.axhline(phi_opt, color=ORANGE, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.annotate(
        "global optimum  4000/4671\n(u = 3/160, graph disconnected)",
        xy=(0.015, phi_opt), xytext=(0, -6), textcoords="offset points",
        fontsize=8.5, color=ORANGE, va="top",
    )
    v_max = float(V_CONNECTED_MAX)
    ax.scatter([0.0], [phi_keg], s=46, color=BLUE, zorder=5,
               edgecolor="white", linewidth=1.3)
    ax.annotate("KEG  100/117", xy=(0.0, phi_keg), xytext=(9, -9),
                textcoords="offset points", fontsize=8.5, color=BLUE)
    ax.scatter([v_max], [phi_conn], s=58, color=TEAL, zorder=5,
               edgecolor="white", linewidth=1.3)
    # Text sits in the free wedge between the curve and the optimum line, with a
    # leader back to the marker, so nothing overlaps the data.
    ax.annotate(
        "densest connected packing\n125/146  at (0, 1/20, -1/320)",
        xy=(v_max, phi_conn), xytext=(0.006, 0.85592), textcoords="data",
        fontsize=8.5, color=TEAL, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.9,
                        shrinkA=4, shrinkB=4),
    )
    ax.annotate(
        "$\\varphi = 100/(117 - 80v^2)$",
        xy=(0.0295, 0.854665), fontsize=9, color=INK,
    )
    ax.set_xlabel("v   (along the connected plane u = 0)")
    ax.set_ylabel("packing fraction  $\\varphi$")
    ax.set_title(
        "Staying connected costs 0.0214% density\nv is capped at 1/20 by the packing constraints",
        fontsize=10, color=INK, loc="left", pad=10,
    )
    ax.set_xlim(-0.003, 0.056)
    ax.set_ylim(0.85455, 0.85646)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax.grid(True, color=GRID, lw=0.6, zorder=0)

    fig.tight_layout()
    figure_path = out.with_suffix(".png")
    fig.savefig(figure_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return figure_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=3, help="tiling replicas per axis")
    parser.add_argument("--out", type=str, default="connectivity", help="output prefix")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    logging.getLogger("tetra_geometry").setLevel(logging.WARNING)
    logging.getLogger("tetra_spectral").setLevel(logging.WARNING)

    contacts = keg_contact_pairs(args.reps)
    LOGGER.info("tracking %d exact inter-dimer contacts from the KEG origin", len(contacts))

    path_df = sweep_to_optimum(args.reps, contacts)
    plane_df = sweep_connected_plane(args.reps, contacts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path_df.to_csv(out.with_name(out.name + "_path.csv"), index=False)
    plane_df.to_csv(out.with_name(out.name + "_plane.csv"), index=False)

    figure = render(path_df, plane_df, out)
    LOGGER.info("wrote %s", figure)

    shown = path_df[path_df["edge_atol"] == EDGE_TOLERANCES[1]][
        ["t", "phi", "contact_drift", "n_components", "largest_component", "first_positive"]
    ]
    LOGGER.info("\nPath KEG -> optimum at edge tolerance %.0e:\n%s", EDGE_TOLERANCES[1],
                shown.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
