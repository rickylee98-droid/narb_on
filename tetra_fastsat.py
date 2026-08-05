"""Optional compiled inner loop for the periodic overlap test.

The separating-axis check runs once per Monte Carlo move on very small arrays --
a handful of tetrahedron pairs at a time.  At that size NumPy's per-call
dispatch overhead dominates the arithmetic completely, so vectorising harder
stops paying: the work is already a rounding error next to the cost of getting
into and out of each ufunc.

This module provides the same computation as one fused, allocation-free loop
compiled with :mod:`numba`.  Two things become available that the array-based
version cannot express:

* **early exit** -- a separating axis proves disjointness, so the moment one is
  found the remaining axes are skipped, and the moment an overlapping pair is
  found the whole configuration is rejected;
* **no temporaries** -- projections are accumulated in registers rather than
  materialised as ``(m, 44, 4)`` arrays.

numba is optional.  When it is missing, :data:`HAVE_NUMBA` is ``False`` and
callers fall back to the NumPy implementation, which is tested to agree.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HAVE_NUMBA", "configuration_is_valid"]

try:  # pragma: no cover - exercised by whichever branch the environment has
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @njit(cache=True, inline="always")
    def _projection_overlap(a, b, ax, ay, az):  # pragma: no cover - compiled
        """Overlap of the two tetrahedra's shadows on the (unnormalised) axis."""
        a_lo = 1.0e308
        a_hi = -1.0e308
        b_lo = 1.0e308
        b_hi = -1.0e308
        for v in range(4):
            pa = a[v, 0] * ax + a[v, 1] * ay + a[v, 2] * az
            if pa < a_lo:
                a_lo = pa
            if pa > a_hi:
                a_hi = pa
            pb = b[v, 0] * ax + b[v, 1] * ay + b[v, 2] * az
            if pb < b_lo:
                b_lo = pb
            if pb > b_hi:
                b_hi = pb
        upper = a_hi if a_hi < b_hi else b_hi
        lower = a_lo if a_lo > b_lo else b_lo
        return upper - lower

    @njit(cache=True)
    def _pair_disjoint(a, b, tolerance):  # pragma: no cover - compiled
        """Separating axis test for one tetrahedron pair, with early exit.

        Face normals are tried first because they separate the overwhelming
        majority of pairs; the 36 edge-pair crosses are only reached when every
        face normal has failed.  Axes are left unnormalised and the tolerance is
        scaled by the axis length instead, which is the same comparison without
        a square root in the common path.
        """
        # Four face normals from each body.  Faces are (1,2,3), (0,2,3),
        # (0,1,3), (0,1,2) -- vertex f omitted from face f.
        for body in range(2):
            for face in range(4):
                i0 = 1 if face == 0 else 0
                i1 = 2 if face <= 1 else 1
                i2 = 3 if face <= 2 else 2
                if body == 0:
                    ux = a[i1, 0] - a[i0, 0]
                    uy = a[i1, 1] - a[i0, 1]
                    uz = a[i1, 2] - a[i0, 2]
                    vx = a[i2, 0] - a[i0, 0]
                    vy = a[i2, 1] - a[i0, 1]
                    vz = a[i2, 2] - a[i0, 2]
                else:
                    ux = b[i1, 0] - b[i0, 0]
                    uy = b[i1, 1] - b[i0, 1]
                    uz = b[i1, 2] - b[i0, 2]
                    vx = b[i2, 0] - b[i0, 0]
                    vy = b[i2, 1] - b[i0, 1]
                    vz = b[i2, 2] - b[i0, 2]
                ax = uy * vz - uz * vy
                ay = uz * vx - ux * vz
                az = ux * vy - uy * vx
                square = ax * ax + ay * ay + az * az
                if square > 1e-24:
                    overlap = _projection_overlap(a, b, ax, ay, az)
                    if overlap <= tolerance * np.sqrt(square):
                        return True

        for ea in range(6):
            if ea == 0:
                p0, p1 = 0, 1
            elif ea == 1:
                p0, p1 = 0, 2
            elif ea == 2:
                p0, p1 = 0, 3
            elif ea == 3:
                p0, p1 = 1, 2
            elif ea == 4:
                p0, p1 = 1, 3
            else:
                p0, p1 = 2, 3
            ux = a[p1, 0] - a[p0, 0]
            uy = a[p1, 1] - a[p0, 1]
            uz = a[p1, 2] - a[p0, 2]
            for eb in range(6):
                if eb == 0:
                    q0, q1 = 0, 1
                elif eb == 1:
                    q0, q1 = 0, 2
                elif eb == 2:
                    q0, q1 = 0, 3
                elif eb == 3:
                    q0, q1 = 1, 2
                elif eb == 4:
                    q0, q1 = 1, 3
                else:
                    q0, q1 = 2, 3
                vx = b[q1, 0] - b[q0, 0]
                vy = b[q1, 1] - b[q0, 1]
                vz = b[q1, 2] - b[q0, 2]
                ax = uy * vz - uz * vy
                ay = uz * vx - ux * vz
                az = ux * vy - uy * vx
                square = ax * ax + ay * ay + az * az
                if square > 1e-24:
                    overlap = _projection_overlap(a, b, ax, ay, az)
                    if overlap <= tolerance * np.sqrt(square):
                        return True
        return False

    @njit(cache=True)
    def _configuration_kernel(  # pragma: no cover - compiled
        verts, lattice, inverse, s0, s1, s2, tolerance, reach
    ):
        """True when no tetrahedron overlaps another, including periodic images.

        Each pair's image box is centred on the integer shift that actually
        brings the two together, so the search never depends on how far apart
        the cell's contents are.
        """
        n = verts.shape[0]
        centres = np.empty((n, 3))
        for i in range(n):
            for d in range(3):
                total = 0.0
                for v in range(4):
                    total += verts[i, v, d]
                centres[i, d] = 0.25 * total

        image = np.empty((4, 3))
        reach_sq = reach * reach

        for i in range(n):
            for j in range(i, n):
                dx = centres[i, 0] - centres[j, 0]
                dy = centres[i, 1] - centres[j, 1]
                dz = centres[i, 2] - centres[j, 2]
                f0 = dx * inverse[0, 0] + dy * inverse[1, 0] + dz * inverse[2, 0]
                f1 = dx * inverse[0, 1] + dy * inverse[1, 1] + dz * inverse[2, 1]
                f2 = dx * inverse[0, 2] + dy * inverse[1, 2] + dz * inverse[2, 2]
                b0 = np.floor(f0 + 0.5)
                b1 = np.floor(f1 + 0.5)
                b2 = np.floor(f2 + 0.5)

                for m0 in range(-s0, s0 + 1):
                    n0 = b0 + m0
                    for m1 in range(-s1, s1 + 1):
                        n1 = b1 + m1
                        for m2 in range(-s2, s2 + 1):
                            n2 = b2 + m2
                            if i == j:
                                # n and -n are the same pair; n = 0 is the
                                # particle against itself.  Keep one half.
                                if not (
                                    n0 > 0
                                    or (n0 == 0 and (n1 > 0 or (n1 == 0 and n2 > 0)))
                                ):
                                    continue
                            sx = (
                                n0 * lattice[0, 0]
                                + n1 * lattice[1, 0]
                                + n2 * lattice[2, 0]
                            )
                            sy = (
                                n0 * lattice[0, 1]
                                + n1 * lattice[1, 1]
                                + n2 * lattice[2, 1]
                            )
                            sz = (
                                n0 * lattice[0, 2]
                                + n1 * lattice[1, 2]
                                + n2 * lattice[2, 2]
                            )
                            rx = dx - sx
                            ry = dy - sy
                            rz = dz - sz
                            if rx * rx + ry * ry + rz * rz >= reach_sq:
                                continue
                            for v in range(4):
                                image[v, 0] = verts[j, v, 0] + sx
                                image[v, 1] = verts[j, v, 1] + sy
                                image[v, 2] = verts[j, v, 2] + sz
                            if not _pair_disjoint(verts[i], image, tolerance):
                                return False
        return True


def configuration_is_valid(
    verts: np.ndarray,
    lattice: np.ndarray,
    inverse: np.ndarray,
    spans: tuple[int, int, int],
    tolerance: float,
    reach: float,
) -> bool:
    """Compiled periodic overlap test.  Requires :data:`HAVE_NUMBA`."""
    if not HAVE_NUMBA:  # pragma: no cover
        raise RuntimeError("numba is not available")
    return bool(
        _configuration_kernel(
            np.ascontiguousarray(verts, dtype=np.float64),
            np.ascontiguousarray(lattice, dtype=np.float64),
            np.ascontiguousarray(inverse, dtype=np.float64),
            int(spans[0]),
            int(spans[1]),
            int(spans[2]),
            float(tolerance),
            float(reach),
        )
    )
