from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plate_fea.mesh import HeterosisMesh


@dataclass(frozen=True)
class PlateWithHoleGeometry:
    outer_width: float = 500.0
    outer_height: float = 300.0
    hole_width: float = 250.0
    hole_height: float = 180.0

    @property
    def hole_x_min(self) -> float:
        return 0.5 * (self.outer_width - self.hole_width)

    @property
    def hole_x_max(self) -> float:
        return 0.5 * (self.outer_width + self.hole_width)

    @property
    def hole_y_min(self) -> float:
        return 0.5 * (self.outer_height - self.hole_height)

    @property
    def hole_y_max(self) -> float:
        return 0.5 * (self.outer_height + self.hole_height)


def _segment_lines(
    start: float,
    end: float,
    n_div: int,
    clustering: str = "none",
    *,
    power: float = 1.8,
) -> np.ndarray:
    if n_div < 1:
        raise ValueError("n_div must be >= 1")
    if not end > start:
        raise ValueError("segment end must be greater than start")
    if power <= 0.0:
        raise ValueError("power must be > 0")

    s = np.linspace(0.0, 1.0, n_div + 1)
    if clustering == "none":
        mapped = s
    elif clustering == "start":
        mapped = s**power
    elif clustering == "end":
        mapped = 1.0 - (1.0 - s) ** power
    elif clustering == "both":
        mapped = 0.5 * (1.0 - np.cos(np.pi * s))
    else:
        raise ValueError("clustering must be one of: none, start, end, both")

    return start + (end - start) * mapped


def _merge_lines(*line_parts: np.ndarray) -> np.ndarray:
    out = [line_parts[0]]
    for part in line_parts[1:]:
        out.append(part[1:])
    merged = np.concatenate(out)
    if np.any(np.diff(merged) <= 0.0):
        raise ValueError("line coordinates must be strictly increasing")
    return merged


def _node_key(x: float, y: float, digits: int = 9) -> tuple[float, float]:
    return (round(x, digits), round(y, digits))


class MeshGenerator(Protocol):
    def generate(self) -> HeterosisMesh: ...


@dataclass(frozen=True)
class UniformEightBlockQ8Generator:
    """
    Robust baseline: uniform divisions per band (outer–hole–outer), no grading.

    This eliminates the persistent "one very thin strip" artifact caused by
    combining block interfaces with graded spacing.

    Parameters
    ----------
    resolution:
        Controls global mesh density (larger => finer everywhere).
    hole_refine:
        Extra refinement on the hole bands only (larger => finer near the hole).
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 2

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")
        if self.resolution < 1:
            raise ValueError("resolution must be >= 1")
        if self.hole_refine < 0:
            raise ValueError("hole_refine must be >= 0")

        # Divisions per region (uniform within each region band).
        n_outer_x = 2 + 2 * self.resolution
        n_outer_y = 2 + 2 * self.resolution
        n_hole_x = 4 + 2 * self.resolution + 2 * self.hole_refine
        n_hole_y = 4 + 2 * self.resolution + 2 * self.hole_refine

        x_left = np.linspace(0.0, g.hole_x_min, n_outer_x + 1)
        x_mid = np.linspace(g.hole_x_min, g.hole_x_max, n_hole_x + 1)
        x_right = np.linspace(g.hole_x_max, g.outer_width, n_outer_x + 1)
        x_lines = np.concatenate([x_left[:-1], x_mid[:-1], x_right])

        y_bot = np.linspace(0.0, g.hole_y_min, n_outer_y + 1)
        y_mid = np.linspace(g.hole_y_min, g.hole_y_max, n_hole_y + 1)
        y_top = np.linspace(g.hole_y_max, g.outer_height, n_outer_y + 1)
        y_lines = np.concatenate([y_bot[:-1], y_mid[:-1], y_top])

        return _build_q8_mesh_from_cartesian_lines(g, x_lines=x_lines, y_lines=y_lines)


@dataclass(frozen=True)
class UniformBufferRingQ8Generator:
    """
    Uniform buffer-ring mesh (no grading, no thin strips).

    Topology: outer -> buffer -> hole band -> buffer -> outer (in both x and y).
    Each segment is divided uniformly with an integer count derived from:
      - resolution: global density
      - hole_refine: extra density near the hole
      - buffer: symmetric ring thickness [mm]
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 2
    buffer: float = 30.0

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")
        if self.resolution < 1:
            raise ValueError("resolution must be >= 1")
        if self.hole_refine < 0:
            raise ValueError("hole_refine must be >= 0")
        if self.buffer <= 0.0:
            raise ValueError("buffer must be > 0")

        # Buffer rectangle must lie strictly between hole and outer boundary.
        buf_x_min = g.hole_x_min - self.buffer
        buf_x_max = g.hole_x_max + self.buffer
        buf_y_min = g.hole_y_min - self.buffer
        buf_y_max = g.hole_y_max + self.buffer
        if not (0.0 < buf_x_min < g.hole_x_min < g.hole_x_max < buf_x_max < g.outer_width):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")
        if not (0.0 < buf_y_min < g.hole_y_min < g.hole_y_max < buf_y_max < g.outer_height):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")

        # Uniform divisions per segment.
        n_outer = 2 + self.resolution
        n_buffer = 2 + self.resolution
        n_hole_x = 6 + 2 * self.resolution + 2 * self.hole_refine
        n_hole_y = 6 + 2 * self.resolution + 2 * self.hole_refine

        x0 = np.linspace(0.0, buf_x_min, n_outer + 1)
        x1 = np.linspace(buf_x_min, g.hole_x_min, n_buffer + 1)
        x2 = np.linspace(g.hole_x_min, g.hole_x_max, n_hole_x + 1)
        x3 = np.linspace(g.hole_x_max, buf_x_max, n_buffer + 1)
        x4 = np.linspace(buf_x_max, g.outer_width, n_outer + 1)
        x_lines = np.concatenate([x0[:-1], x1[:-1], x2[:-1], x3[:-1], x4])

        y0 = np.linspace(0.0, buf_y_min, n_outer + 1)
        y1 = np.linspace(buf_y_min, g.hole_y_min, n_buffer + 1)
        y2 = np.linspace(g.hole_y_min, g.hole_y_max, n_hole_y + 1)
        y3 = np.linspace(g.hole_y_max, buf_y_max, n_buffer + 1)
        y4 = np.linspace(buf_y_max, g.outer_height, n_outer + 1)
        y_lines = np.concatenate([y0[:-1], y1[:-1], y2[:-1], y3[:-1], y4])

        return _build_q8_mesh_from_cartesian_lines(g, x_lines=x_lines, y_lines=y_lines)


@dataclass(frozen=True)
class EightBlockStructuredQ8Generator:
    """
    Baseline (your current) approach: 8-block rectilinear structured mesh.

    Still useful as a reference mesh for convergence studies, but not the only option.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    n_left: int = 6
    n_middle_x: int = 12
    n_right: int = 6
    n_bottom: int = 4
    n_middle_y: int = 10
    n_top: int = 6
    grading_power: float = 1.05

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")

        x_lines = _merge_lines(
            _segment_lines(0.0, g.hole_x_min, self.n_left, clustering="end", power=self.grading_power),
            _segment_lines(g.hole_x_min, g.hole_x_max, self.n_middle_x, clustering="both"),
            _segment_lines(g.hole_x_max, g.outer_width, self.n_right, clustering="start", power=self.grading_power),
        )
        y_lines = _merge_lines(
            _segment_lines(0.0, g.hole_y_min, self.n_bottom, clustering="end", power=self.grading_power),
            _segment_lines(g.hole_y_min, g.hole_y_max, self.n_middle_y, clustering="both"),
            _segment_lines(g.hole_y_max, g.outer_height, self.n_top, clustering="start", power=self.grading_power),
        )

        return _build_q8_mesh_from_cartesian_lines(g, x_lines=x_lines, y_lines=y_lines)


@dataclass(frozen=True)
class BufferRingStructuredQ8Generator:
    """
    Improved structured mesh: adds a "buffer rectangle" around the hole.

    This keeps clean mapped quads while allowing smoother grading and more control
    around the loaded top inner edge and the hole corners.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    buffer: float = 30.0

    # segment divisions (outer -> buffer -> hole band -> buffer -> outer)
    n_outer_left: int = 4
    n_buffer_left: int = 3
    n_hole_band_x: int = 14
    n_buffer_right: int = 3
    n_outer_right: int = 4

    n_outer_bottom: int = 3
    n_buffer_bottom: int = 3
    n_hole_band_y: int = 12
    n_buffer_top: int = 5
    n_outer_top: int = 4
    grading_power: float = 1.10

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")

        if self.buffer <= 0.0:
            raise ValueError("buffer must be > 0")

        buf_x_min = max(0.0, g.hole_x_min - self.buffer)
        buf_x_max = min(g.outer_width, g.hole_x_max + self.buffer)
        buf_y_min = max(0.0, g.hole_y_min - self.buffer)
        buf_y_max = min(g.outer_height, g.hole_y_max + self.buffer)
        if not (0.0 < buf_x_min < g.hole_x_min < g.hole_x_max < buf_x_max < g.outer_width):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")
        if not (0.0 < buf_y_min < g.hole_y_min < g.hole_y_max < buf_y_max < g.outer_height):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")

        # Bias more resolution toward the HOLE and especially toward the TOP of the hole (load application region).
        x_lines = _merge_lines(
            _segment_lines(0.0, buf_x_min, self.n_outer_left, clustering="end", power=self.grading_power),
            _segment_lines(buf_x_min, g.hole_x_min, self.n_buffer_left, clustering="end", power=self.grading_power),
            _segment_lines(g.hole_x_min, g.hole_x_max, self.n_hole_band_x, clustering="both"),
            _segment_lines(g.hole_x_max, buf_x_max, self.n_buffer_right, clustering="start", power=self.grading_power),
            _segment_lines(buf_x_max, g.outer_width, self.n_outer_right, clustering="start", power=self.grading_power),
        )
        y_lines = _merge_lines(
            _segment_lines(0.0, buf_y_min, self.n_outer_bottom, clustering="end", power=self.grading_power),
            _segment_lines(buf_y_min, g.hole_y_min, self.n_buffer_bottom, clustering="end", power=self.grading_power),
            _segment_lines(g.hole_y_min, g.hole_y_max, self.n_hole_band_y, clustering="both"),
            _segment_lines(g.hole_y_max, buf_y_max, self.n_buffer_top, clustering="start", power=self.grading_power),
            _segment_lines(buf_y_max, g.outer_height, self.n_outer_top, clustering="start", power=self.grading_power),
        )

        return _build_q8_mesh_from_cartesian_lines(g, x_lines=x_lines, y_lines=y_lines)


@dataclass(frozen=True)
class WarpedInteriorQ8Generator:
    """
    Make elements non-orthogonal (non-90°) while keeping the outer boundary and hole exact.

    This is useful to experiment with 'smarter' meshes that bias directions (e.g. toward
    clamped edges or toward the loaded hole top edge) without changing topology.
    """

    base: MeshGenerator
    amplitude: float = 18.0
    p: float = 1.6

    def generate(self) -> HeterosisMesh:
        mesh = self.base.generate()
        xy = mesh.node_coordinates.copy()

        # Fixed boundaries: outer rectangle and the inner hole rectangle.
        # We warp only interior nodes with a mask that goes to zero at both boundaries.
        # This intentionally creates non-90° quads (skew) while preserving the exact edges.
        g = _infer_geometry_from_mesh(mesh)
        x = xy[:, 0]
        y = xy[:, 1]

        dist_outer = np.minimum.reduce([x - 0.0, g.outer_width - x, y - 0.0, g.outer_height - y])
        dist_hole = _distance_to_axis_aligned_rectangle(
            x,
            y,
            x_min=g.hole_x_min,
            x_max=g.hole_x_max,
            y_min=g.hole_y_min,
            y_max=g.hole_y_max,
        )

        eps = 1e-12
        max_outer = float(np.max(dist_outer) + eps)
        max_hole = float(np.max(dist_hole) + eps)
        mask = (np.clip(dist_outer / max_outer, 0.0, 1.0) ** self.p) * (np.clip(dist_hole / max_hole, 0.0, 1.0) ** self.p)

        xc = 0.5 * g.outer_width
        yc = 0.5 * g.outer_height

        # Skew field: biases toward top-left (common clamped region in your description)
        # while being exactly zero on boundaries via mask.
        dx = (y - yc) / max(g.outer_height, eps)
        dy = (x - xc) / max(g.outer_width, eps)
        x_warp = x + self.amplitude * mask * dx
        y_warp = y - 0.65 * self.amplitude * mask * dy

        warped_node_coordinates = np.column_stack([x_warp, y_warp])
        return HeterosisMesh.from_arrays(
            node_coordinates=warped_node_coordinates,
            w_location_matrix=mesh.w_location_matrix,
            theta_location_matrix=mesh.theta_location_matrix,
        )


def _distance_to_axis_aligned_rectangle(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    dx = np.maximum.reduce([x_min - x, np.zeros_like(x), x - x_max])
    dy = np.maximum.reduce([y_min - y, np.zeros_like(y), y - y_max])
    return np.sqrt(dx**2 + dy**2)


def _infer_geometry_from_mesh(mesh: HeterosisMesh) -> PlateWithHoleGeometry:
    # Mesh was generated from axis-aligned rectangles; infer from coordinates.
    x = mesh.node_coordinates[:, 0]
    y = mesh.node_coordinates[:, 1]
    outer_width = float(np.max(x) - np.min(x))
    outer_height = float(np.max(y) - np.min(y))

    # The hole boundary nodes are those that have a near-zero distance to the hole rectangle;
    # however we don't know the hole rectangle a priori. Infer it from missing elements is hard,
    # so we store geometry via generator and only use this for WarpedInteriorQ8Generator,
    # where the base generator is expected to be one of the structured rectangle-based ones.
    #
    # We reconstruct the hole rectangle from the most common interior "gaps":
    # take the set of unique x,y coordinates and find the largest gap around the center.
    ux = np.unique(np.round(x, 9))
    uy = np.unique(np.round(y, 9))
    xc = np.min(ux) + 0.5 * outer_width
    yc = np.min(uy) + 0.5 * outer_height

    def find_gap(u: np.ndarray, center: float) -> tuple[float, float]:
        dif = np.diff(u)
        # look for the largest gap nearest the center
        gap_idx = int(np.argmax(dif))
        # fallback: choose gap whose midpoint is closest to center
        midpoints = 0.5 * (u[:-1] + u[1:])
        gap_idx = int(np.argmin(np.abs(midpoints - center)))
        return float(u[gap_idx]), float(u[gap_idx + 1])

    hole_x_min, hole_x_max = find_gap(ux, xc)
    hole_y_min, hole_y_max = find_gap(uy, yc)

    hole_width = hole_x_max - hole_x_min
    hole_height = hole_y_max - hole_y_min
    return PlateWithHoleGeometry(
        outer_width=outer_width,
        outer_height=outer_height,
        hole_width=hole_width,
        hole_height=hole_height,
    )


def _build_q8_mesh_from_cartesian_lines(
    geometry: PlateWithHoleGeometry,
    *,
    x_lines: np.ndarray,
    y_lines: np.ndarray,
) -> HeterosisMesh:
    nodes: list[tuple[float, float]] = []
    node_ids: dict[tuple[float, float], int] = {}
    elements: list[list[int]] = []

    def get_node_id(x: float, y: float) -> int:
        key = _node_key(x, y)
        if key not in node_ids:
            node_ids[key] = len(nodes)
            nodes.append((x, y))
        return node_ids[key]

    for ix in range(len(x_lines) - 1):
        x0 = float(x_lines[ix])
        x1 = float(x_lines[ix + 1])
        x_mid = 0.5 * (x0 + x1)

        in_middle_x = (x0 >= geometry.hole_x_min) and (x1 <= geometry.hole_x_max)
        for iy in range(len(y_lines) - 1):
            y0 = float(y_lines[iy])
            y1 = float(y_lines[iy + 1])
            y_mid = 0.5 * (y0 + y1)

            in_middle_y = (y0 >= geometry.hole_y_min) and (y1 <= geometry.hole_y_max)
            if in_middle_x and in_middle_y:
                continue

            local = [
                get_node_id(x0, y0),
                get_node_id(x1, y0),
                get_node_id(x1, y1),
                get_node_id(x0, y1),
                get_node_id(x_mid, y0),
                get_node_id(x1, y_mid),
                get_node_id(x_mid, y1),
                get_node_id(x0, y_mid),
            ]
            elements.append(local)

    node_coordinates = np.asarray(nodes, dtype=float)
    w_location_matrix = np.asarray(elements, dtype=int).T
    return HeterosisMesh.from_arrays(node_coordinates=node_coordinates, w_location_matrix=w_location_matrix)


def generate_structured_q8_plate_with_hole_mesh(
    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry(),
    n_left: int = 6,
    n_middle_x: int = 12,
    n_right: int = 6,
    n_bottom: int = 4,
    n_middle_y: int = 10,
    n_top: int = 6,
) -> HeterosisMesh:
    """
    Backwards-compatible wrapper: EightBlockStructuredQ8Generator(...).generate().
    """
    return EightBlockStructuredQ8Generator(
        geometry=geometry,
        n_left=n_left,
        n_middle_x=n_middle_x,
        n_right=n_right,
        n_bottom=n_bottom,
        n_middle_y=n_middle_y,
        n_top=n_top,
    ).generate()


def generate_rectangular_heterosis_mesh(width: float, height: float, nx: int, ny: int) -> HeterosisMesh:
    """
    Generate a structured heterosis mesh for a full rectangle (no hole).

    Parameters
    ----------
    width, height:
        Rectangle dimensions.
    nx, ny:
        Number of Q8 elements along x and y.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    # Base Q8 grid for w/geometry; theta Q9 center nodes are auto-generated by HeterosisMesh.from_arrays.
    i_max = 2 * nx
    j_max = 2 * ny
    x_grid = np.linspace(0.0, width, i_max + 1)
    y_grid = np.linspace(0.0, height, j_max + 1)

    node_id_map: dict[tuple[int, int], int] = {}
    node_coordinates: list[list[float]] = []

    for j in range(j_max + 1):
        for i in range(i_max + 1):
            # Exclude Q8 center points (odd, odd).
            if (i % 2 == 1) and (j % 2 == 1):
                continue
            node_id_map[(i, j)] = len(node_coordinates)
            node_coordinates.append([float(x_grid[i]), float(y_grid[j])])

    elements: list[list[int]] = []
    for ey in range(ny):
        for ex in range(nx):
            i0 = 2 * ex
            j0 = 2 * ey
            # Local Q8 order: bl, br, tr, tl, mid-bottom, mid-right, mid-top, mid-left
            local_keys = [
                (i0, j0),
                (i0 + 2, j0),
                (i0 + 2, j0 + 2),
                (i0, j0 + 2),
                (i0 + 1, j0),
                (i0 + 2, j0 + 1),
                (i0 + 1, j0 + 2),
                (i0, j0 + 1),
            ]
            elements.append([node_id_map[k] for k in local_keys])

    node_coordinates_arr = np.asarray(node_coordinates, dtype=float)
    w_location_matrix = np.asarray(elements, dtype=int).T
    return HeterosisMesh.from_arrays(node_coordinates=node_coordinates_arr, w_location_matrix=w_location_matrix)


def generate_rectangular_q8_mesh(width: float, height: float, nx: int, ny: int) -> HeterosisMesh:
    """Backward-compatible alias. Prefer generate_rectangular_heterosis_mesh."""
    return generate_rectangular_heterosis_mesh(width=width, height=height, nx=nx, ny=ny)
