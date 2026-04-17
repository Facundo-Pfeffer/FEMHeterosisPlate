"""Heterosis mesh generators (Q8 displacement + Q9 rotation interpolation)."""

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
    Structured heterosis mesh with uniform outer-hole-outer spacing.

    Parameters:
        resolution: Global density level (larger -> finer mesh).
        hole_refine: Additional density in hole-adjacent bands.
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
    Structured heterosis mesh with a uniform buffer band around the hole.

    Parameters:
        resolution: Global density level (larger -> finer mesh).
        hole_refine: Additional density in hole-adjacent bands.
        buffer: Buffer-band thickness around the hole boundary.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 2
    buffer: float = 30.0

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")
        if self.resolution < -1:
            raise ValueError("resolution must be >= -1")
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
    Eight-block structured heterosis mesh kept for compatibility and comparisons.

    Parameters:
        n_left, n_middle_x, n_right: Element divisions in x-direction block bands.
        n_bottom, n_middle_y, n_top: Element divisions in y-direction block bands.
        grading_power: Clustering exponent for graded segments.
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
    Structured heterosis mesh with a buffer rectangle and graded spacing.

    Parameters:
        buffer: Offset between hole boundary and buffer rectangle.
        grading_power: Clustering exponent for graded segments.
        n_*: Element divisions for each outer/buffer/hole segment in x and y.
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
class GradedBoundarySensitiveQ8Generator:
    """
    Graded heterosis mesh refined near load and BC-transition regions.

    Parameters:
        resolution: Global density level (larger -> finer mesh).
        hole_refine: Additional density in hole-adjacent bands.
        buffer: Buffer-band thickness around the hole.
        grading_power: Clustering exponent for directional grading.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 3
    buffer: float = 30.0
    grading_power: float = 1.35

    def generate(self) -> HeterosisMesh:
        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")
        if self.resolution < -1:
            raise ValueError("resolution must be >= -1")
        if self.hole_refine < 0:
            raise ValueError("hole_refine must be >= 0")
        if self.buffer <= 0.0:
            raise ValueError("buffer must be > 0")
        if self.grading_power <= 0.0:
            raise ValueError("grading_power must be > 0")

        buf_x_min = g.hole_x_min - self.buffer
        buf_x_max = g.hole_x_max + self.buffer
        buf_y_min = g.hole_y_min - self.buffer
        buf_y_max = g.hole_y_max + self.buffer
        if not (0.0 < buf_x_min < g.hole_x_min < g.hole_x_max < buf_x_max < g.outer_width):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")
        if not (0.0 < buf_y_min < g.hole_y_min < g.hole_y_max < buf_y_max < g.outer_height):
            raise ValueError("buffer too large or too small; symmetric buffer must lie between hole and outer boundary")

        n_outer = 2 + self.resolution
        n_buffer = 2 + self.resolution
        # Increase divisions in the hole-adjacent band to capture load-transfer gradients.
        n_hole_x = 7 + 2 * self.resolution + 2 * self.hole_refine
        n_hole_y = 9 + 2 * self.resolution + 2 * self.hole_refine

        p = self.grading_power
        x_lines = _merge_lines(
            # Refine near the x=0 outer boundary (near the bottom-left transition region).
            _segment_lines(0.0, buf_x_min, n_outer + 1, clustering="start", power=p),
            _segment_lines(buf_x_min, g.hole_x_min, n_buffer + 1, clustering="start", power=p),
            _segment_lines(g.hole_x_min, g.hole_x_max, n_hole_x, clustering="both"),
            _segment_lines(g.hole_x_max, buf_x_max, n_buffer + 1, clustering="end", power=p),
            # Refine near the x=outer_width outer boundary (near the top-right transition region).
            _segment_lines(buf_x_max, g.outer_width, n_outer + 1, clustering="end", power=p),
        )
        y_lines = _merge_lines(
            # Refine toward y=0 at the bottom-left transition region.
            _segment_lines(0.0, buf_y_min, n_outer + 1, clustering="start", power=p),
            _segment_lines(buf_y_min, g.hole_y_min, n_buffer + 1, clustering="start", power=p),
            # Refine toward the loaded inner boundary segment (y = hole_y_max).
            _segment_lines(g.hole_y_min, g.hole_y_max, n_hole_y, clustering="end", power=max(1.05, p)),
            _segment_lines(g.hole_y_max, buf_y_max, n_buffer + 1, clustering="start", power=p),
            # Refine toward y=outer_height at the top-right transition region.
            _segment_lines(buf_y_max, g.outer_height, n_outer + 2, clustering="end", power=p),
        )
        return _build_q8_mesh_from_cartesian_lines(g, x_lines=x_lines, y_lines=y_lines)


@dataclass(frozen=True)
class WarpedInteriorQ8Generator:
    """
    Applies an interior coordinate mapping while preserving physical boundaries.

    Parameters:
        base: Base mesh generator that defines boundary-conforming connectivity.
        amplitude: Displacement magnitude of the coordinate mapping.
        p: Boundary-decay exponent for the interior mask.
    """

    base: MeshGenerator
    amplitude: float = 18.0
    p: float = 1.6

    def generate(self) -> HeterosisMesh:
        mesh = self.base.generate()
        xy = mesh.node_coordinates.copy()

        # Keep outer and inner boundaries fixed.
        # Apply the mapping to interior nodes only (mask is zero on both boundaries).
        # This changes element distortion/skewness without changing domain boundaries.
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

        # Directional mapping field with zero displacement on boundaries.
        dx = (y - yc) / max(g.outer_height, eps)
        dy = (x - xc) / max(g.outer_width, eps)
        x_warp = x + self.amplitude * mask * dx
        y_warp = y - 0.65 * self.amplitude * mask * dy

        warped_node_coordinates = np.column_stack([x_warp, y_warp])
        return HeterosisMesh.from_arrays(
            node_coordinates=warped_node_coordinates,
            w_location_matrix=mesh.w_location_matrix,
        )


@dataclass(frozen=True)
class FocusedWarpedInteriorQ8Generator:
    """
    Interior coordinate mapping focused near load and BC-transition regions.

    Parameters:
        base: Base mesh generator that defines mesh topology/connectivity.
        geometry: Plate-with-hole geometry used for weighting regions.
        amplitude: Displacement magnitude of the coordinate mapping.
        p: Boundary-decay exponent for the interior mask.
    """

    base: MeshGenerator
    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    amplitude: float = 60.0
    p: float = 1.45

    def generate(self) -> HeterosisMesh:
        mesh = self.base.generate()
        g = self.geometry
        xy = mesh.node_coordinates.copy()
        x = xy[:, 0]
        y = xy[:, 1]

        # Zero displacement on hole and outer boundaries.
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

        def gaussian(xc: float, yc: float, sx: float, sy: float) -> np.ndarray:
            return np.exp(-0.5 * (((x - xc) / sx) ** 2 + ((y - yc) / sy) ** 2))

        # Weight field centers:
        # - loaded inner boundary segment (top edge of the hole)
        # - outer boundary-condition transition regions (bottom-left and top-right corners)
        load_w = gaussian(
            0.5 * (g.hole_x_min + g.hole_x_max),
            g.hole_y_max + 0.15 * (g.outer_height - g.hole_y_max),
            sx=0.22 * g.outer_width,
            sy=0.16 * g.outer_height,
        )
        bl_w = gaussian(0.12 * g.outer_width, 0.12 * g.outer_height, sx=0.16 * g.outer_width, sy=0.16 * g.outer_height)
        tr_w = gaussian(0.88 * g.outer_width, 0.88 * g.outer_height, sx=0.16 * g.outer_width, sy=0.16 * g.outer_height)
        field_w = np.clip(1.00 * load_w + 0.80 * bl_w + 0.80 * tr_w, 0.0, None)

        # Curl-like mapping increases element skewness while remaining smooth for moderate amplitude.
        u = (y - 0.5 * g.outer_height) / max(g.outer_height, eps)
        v = (x - 0.5 * g.outer_width) / max(g.outer_width, eps)
        x_warp = x + self.amplitude * mask * field_w * (0.95 * u + 0.25 * v)
        y_warp = y - 0.80 * self.amplitude * mask * field_w * (0.90 * v - 0.20 * u)

        warped_xy = np.column_stack([x_warp, y_warp])
        return HeterosisMesh.from_arrays(
            node_coordinates=warped_xy,
            w_location_matrix=mesh.w_location_matrix,
        )


def _gmsh_bc_transition_sample_coordinates(
    g: PlateWithHoleGeometry,
    clamped_outer_edges: tuple[str, ...],
    *,
    sample_count: int,
    transition_fraction: float,
) -> list[tuple[float, float]]:
    """
    Return geometric sampling coordinates on short outer-boundary segments at **mixed**
    clamped/free corners (each corner uses the two edges meeting there).

    These are geometric coordinates used to define Gmsh distance fields. They are not FE mesh nodes.
    """
    valid = frozenset({"left", "right", "bottom", "top"})
    clamped = {e for e in clamped_outer_edges if e in valid}
    free = valid - clamped
    outer_width, outer_height = g.outer_width, g.outer_height
    sample_coordinates: list[tuple[float, float]] = []
    transition_fraction_clamped = float(np.clip(transition_fraction, 0.0, 1.0))
    sample_count = max(2, int(sample_count))

    # TL (0,H): left / top
    if "left" in clamped and "top" in free:
        for y in np.linspace((1.0 - transition_fraction_clamped) * outer_height, outer_height, sample_count):
            sample_coordinates.append((0.0, float(y)))
    if "top" in clamped and "left" in free:
        for x in np.linspace(0.0, transition_fraction_clamped * outer_width, sample_count):
            sample_coordinates.append((float(x), float(outer_height)))

    # TR (W,H): right / top — default: top clamped, right free → vertical right edge, upper segment
    if "right" in clamped and "top" in free:
        for x in np.linspace((1.0 - transition_fraction_clamped) * outer_width, outer_width, sample_count):
            sample_coordinates.append((float(x), float(outer_height)))
    if "top" in clamped and "right" in free:
        for y in np.linspace((1.0 - transition_fraction_clamped) * outer_height, outer_height, sample_count):
            sample_coordinates.append((float(outer_width), float(y)))

    # BL (0,0): left / bottom
    if "left" in clamped and "bottom" in free:
        for y in np.linspace(0.0, transition_fraction_clamped * outer_height, sample_count):
            sample_coordinates.append((0.0, float(y)))
    if "bottom" in clamped and "left" in free:
        for x in np.linspace(0.0, transition_fraction_clamped * outer_width, sample_count):
            sample_coordinates.append((float(x), 0.0))

    # BR (W,0): right / bottom
    if "right" in clamped and "bottom" in free:
        for y in np.linspace(0.0, transition_fraction_clamped * outer_height, sample_count):
            sample_coordinates.append((float(outer_width), float(y)))
    if "bottom" in clamped and "right" in free:
        for x in np.linspace((1.0 - transition_fraction_clamped) * outer_width, outer_width, sample_count):
            sample_coordinates.append((float(x), 0.0))

    # Remove duplicates at shared segment endpoints.
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[float, float]] = []
    scale = 1e7
    for a, b in sample_coordinates:
        key = (int(round(a * scale)), int(round(b * scale)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b))
    return deduped


@dataclass(frozen=True)
class GmshBoundarySensitiveQ8Generator:
    """
    Gmsh-based heterosis mesh with distance-field sizing near selected boundaries.

    Parameters:
        resolution: Global density level (larger -> finer mesh).
        hole_refine: Additional refinement near hole and transition regions.
        clamped_outer_edges: Outer boundary segments with clamped boundary conditions.
        bc_transition_fraction: Unused (reserved); gmsh sizing uses outer corner points only, not edge polylines.
        bc_transition_samples: Unused (reserved); see bc_transition_fraction.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 2
    clamped_outer_edges: tuple[str, ...] = ("left", "top")
    bc_transition_fraction: float = 0.45
    bc_transition_samples: int = 40

    def generate(self) -> HeterosisMesh:
        try:
            import gmsh  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "gmsh runtime unavailable. Install Python package with `pip install --upgrade gmsh` "
                "and ensure system OpenGL dependency `libGLU.so.1` is installed (e.g. `apt install libglu1-mesa`)."
            ) from exc

        g = self.geometry
        if g.hole_width >= g.outer_width or g.hole_height >= g.outer_height:
            raise ValueError("hole dimensions must be strictly smaller than outer dimensions")
        if self.resolution < -1:
            raise ValueError("resolution must be >= -1")
        if self.hole_refine < 0:
            raise ValueError("hole_refine must be >= 0")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_with_hole_q8")
        occ = gmsh.model.occ

        try:
            # Outer plate and centered hole rectangles.
            outer = occ.addRectangle(0.0, 0.0, 0.0, g.outer_width, g.outer_height)
            hole = occ.addRectangle(g.hole_x_min, g.hole_y_min, 0.0, g.hole_width, g.hole_height)
            cut, _ = occ.cut([(2, outer)], [(2, hole)], removeObject=True, removeTool=True)
            occ.synchronize()
            if len(cut) != 1:
                raise RuntimeError("Unexpected gmsh boolean-cut result for plate-with-hole.")
            surface_tag = int(cut[0][1])

            # Base sizes: near features smaller, far field larger.
            res_scale = max(0.2, 1.0 + 0.22 * float(self.resolution))
            # Coarsest preset (resolution == -1): slightly larger far-field size so min mesh count
            # can drop below the res>=0 ladder without changing higher-resolution behaviour.
            far_divisor = 7.0 if self.resolution == -1 else 8.5
            lc_far = min(g.outer_width, g.outer_height) / (far_divisor * res_scale)
            lc_hole = lc_far / (2.0 + 0.38 * float(self.hole_refine))
            lc_corner = lc_far / (2.4 + 0.30 * float(self.hole_refine))

            # Gather boundary entities.
            boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
            outer_curves: list[int] = []
            hole_curves: list[int] = []
            hole_top_curves: list[int] = []
            for dim, tag in boundary_curves:
                if dim != 1:
                    continue
                curve_tag = int(tag)
                pts = gmsh.model.getBoundary([(1, curve_tag)], oriented=False)
                xy = []
                for _, pt_tag in pts:
                    x, y, _ = gmsh.model.getValue(0, int(pt_tag), [])
                    xy.append((float(x), float(y)))
                ys = [p[1] for p in xy]
                xs = [p[0] for p in xy]
                is_hole_curve = (
                    min(xs) >= g.hole_x_min - 1e-9
                    and max(xs) <= g.hole_x_max + 1e-9
                    and min(ys) >= g.hole_y_min - 1e-9
                    and max(ys) <= g.hole_y_max + 1e-9
                )
                if is_hole_curve:
                    hole_curves.append(curve_tag)
                    if max(abs(y - g.hole_y_max) for y in ys) <= 1e-9:
                        hole_top_curves.append(curve_tag)
                else:
                    outer_curves.append(curve_tag)

            hole_top_set = set(hole_top_curves)
            hole_non_top_curves = [c for c in hole_curves if c not in hole_top_set]
            # Relaxed size at hole corners only (not full hole edges; full curves caused side bands).
            lc_hole_relaxed = float(min(lc_far * 0.90, lc_hole * 1.65))

            p_bl = occ.addPoint(0.0, 0.0, 0.0)
            p_tr = occ.addPoint(g.outer_width, g.outer_height, 0.0)
            p_load = occ.addPoint(0.5 * (g.hole_x_min + g.hole_x_max), g.hole_y_max, 0.0)
            # Mesh-size probes only (not part of the boundary loop): hole rectangle corners.
            p_hole_sw = occ.addPoint(g.hole_x_min, g.hole_y_min, 0.0)
            p_hole_se = occ.addPoint(g.hole_x_max, g.hole_y_min, 0.0)
            p_hole_nw = occ.addPoint(g.hole_x_min, g.hole_y_max, 0.0)
            p_hole_ne = occ.addPoint(g.hole_x_max, g.hole_y_max, 0.0)
            occ.synchronize()

            # Hole boundary: fine threshold on top (load); corner-only sizing elsewhere on the hole.
            field_ids: list[int] = []
            if hole_top_curves:
                f_hole_top_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_hole_top_dist, "CurvesList", hole_top_curves)
                gmsh.model.mesh.field.setNumber(f_hole_top_dist, "Sampling", 80)
                f_hole_top = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_hole_top, "InField", f_hole_top_dist)
                gmsh.model.mesh.field.setNumber(f_hole_top, "SizeMin", lc_hole)
                gmsh.model.mesh.field.setNumber(f_hole_top, "SizeMax", lc_far)
                gmsh.model.mesh.field.setNumber(f_hole_top, "DistMin", 18.0)
                gmsh.model.mesh.field.setNumber(f_hole_top, "DistMax", 70.0)
                field_ids.append(f_hole_top)
            if hole_non_top_curves:
                f_hole_corner_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    f_hole_corner_dist,
                    "PointsList",
                    [int(p_hole_sw), int(p_hole_se), int(p_hole_nw), int(p_hole_ne)],
                )
                f_hole_corner = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_hole_corner, "InField", f_hole_corner_dist)
                gmsh.model.mesh.field.setNumber(f_hole_corner, "SizeMin", lc_hole_relaxed)
                gmsh.model.mesh.field.setNumber(f_hole_corner, "SizeMax", lc_far)
                gmsh.model.mesh.field.setNumber(f_hole_corner, "DistMin", 8.0)
                gmsh.model.mesh.field.setNumber(f_hole_corner, "DistMax", 28.0)
                field_ids.append(f_hole_corner)
            if not hole_top_curves and not hole_non_top_curves:
                f_hole_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_hole_dist, "CurvesList", hole_curves)
                gmsh.model.mesh.field.setNumber(f_hole_dist, "Sampling", 80)
                f_hole = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_hole, "InField", f_hole_dist)
                gmsh.model.mesh.field.setNumber(f_hole, "SizeMin", lc_hole)
                gmsh.model.mesh.field.setNumber(f_hole, "SizeMax", lc_far)
                gmsh.model.mesh.field.setNumber(f_hole, "DistMin", 18.0)
                gmsh.model.mesh.field.setNumber(f_hole, "DistMax", 70.0)
                field_ids.append(f_hole)

            # Extra bias near load zone (hole-top region).
            f_load_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_load_dist, "CurvesList", hole_top_curves or hole_curves)
            gmsh.model.mesh.field.setNumbers(f_load_dist, "PointsList", [int(p_load)])
            gmsh.model.mesh.field.setNumber(f_load_dist, "Sampling", 80)
            f_load = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_load, "InField", f_load_dist)
            gmsh.model.mesh.field.setNumber(f_load, "SizeMin", min(lc_hole, 0.9 * lc_corner))
            gmsh.model.mesh.field.setNumber(f_load, "SizeMax", lc_far)
            gmsh.model.mesh.field.setNumber(f_load, "DistMin", 10.0)
            gmsh.model.mesh.field.setNumber(f_load, "DistMax", 55.0)

            # Outer mixed BC corners only (no polylines along edges — those produced full-height side bands).
            f_corner_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_corner_dist, "PointsList", [int(p_bl), int(p_tr)])
            f_corner = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_corner, "InField", f_corner_dist)
            gmsh.model.mesh.field.setNumber(f_corner, "SizeMin", lc_corner)
            gmsh.model.mesh.field.setNumber(f_corner, "SizeMax", lc_far)
            gmsh.model.mesh.field.setNumber(f_corner, "DistMin", 10.0)
            gmsh.model.mesh.field.setNumber(f_corner, "DistMax", 38.0)

            field_ids.extend([f_load, f_corner])

            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

            min_mesh = min(lc_hole, lc_hole_relaxed, lc_corner)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_mesh * 0.70)
            gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far * 1.10)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay option for quad recombination
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            # Keep 8-node quadrilateral geometry nodes; rotational DOFs are handled in HeterosisMesh.
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)

            gmsh.model.mesh.generate(2)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            if len(node_tags) == 0:
                raise RuntimeError("gmsh produced no nodes.")
            xy_all = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
            tag_to_xy = {int(t): (float(xy[0]), float(xy[1])) for t, xy in zip(node_tags, xy_all, strict=True)}

            elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2, tag=surface_tag)
            quad8_elements: list[list[int]] = []
            for etype, node_list in zip(elem_types, elem_node_tags, strict=True):
                name, _, _, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(int(etype))
                if ("Quadrilateral" not in name) or (int(n_nodes) != 8):
                    continue
                arr = np.asarray(node_list, dtype=np.int64).reshape(-1, 8)
                quad8_elements.extend(arr.tolist())
            if not quad8_elements:
                raise RuntimeError("gmsh did not generate any 8-node quadrilateral elements.")

            used_node_tags = sorted({int(t) for elem in quad8_elements for t in elem})
            tag_to_id = {tag: i for i, tag in enumerate(used_node_tags)}
            node_coordinates = np.asarray([tag_to_xy[tag] for tag in used_node_tags], dtype=float)
            w_location_matrix = np.asarray([[tag_to_id[int(t)] for t in elem] for elem in quad8_elements], dtype=int).T
            return HeterosisMesh.from_arrays(node_coordinates=node_coordinates, w_location_matrix=w_location_matrix)
        finally:
            gmsh.finalize()


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
    """Backward-compatible wrapper around `EightBlockStructuredQ8Generator`."""
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
    """Generate a structured heterosis mesh for a full rectangle."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    # Build geometry nodes on the Q8 pattern; Q9-style rotational interpolation nodes are added internally.
    i_max = 2 * nx
    j_max = 2 * ny
    x_grid = np.linspace(0.0, width, i_max + 1)
    y_grid = np.linspace(0.0, height, j_max + 1)

    node_id_map: dict[tuple[int, int], int] = {}
    node_coordinates: list[list[float]] = []

    for j in range(j_max + 1):
        for i in range(i_max + 1):
            # Exclude element center geometry nodes (odd, odd) from the displacement-node layout.
            if (i % 2 == 1) and (j % 2 == 1):
                continue
            node_id_map[(i, j)] = len(node_coordinates)
            node_coordinates.append([float(x_grid[i]), float(y_grid[j])])

    elements: list[list[int]] = []
    for ey in range(ny):
        for ex in range(nx):
            i0 = 2 * ex
            j0 = 2 * ey
            # Local displacement-node order: bl, br, tr, tl, mid-bottom, mid-right, mid-top, mid-left.
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



@dataclass(frozen=True)
class TargetAwareWarpedQ8Generator:
    """
    Conforming graded mesh with interior mapping focused on benchmark critical regions.

    Parameters:
        resolution: Global density level (larger -> finer mesh).
        hole_refine: Additional density near hole-adjacent regions.
        buffer: Buffer-band thickness around the hole.
        grading_power: Clustering exponent for the graded base mesh.
        amplitude: Displacement magnitude of the interior coordinate mapping.
        p: Boundary-decay exponent for the interior mask.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 3
    hole_refine: int = 4
    buffer: float = 25.0
    grading_power: float = 1.35
    amplitude: float = 24.0
    p: float = 1.45

    def generate(self) -> HeterosisMesh:
        g = self.geometry

        base_mesh = GradedBoundarySensitiveQ8Generator(
            geometry=g,
            resolution=self.resolution,
            hole_refine=self.hole_refine,
            buffer=self.buffer,
            grading_power=self.grading_power,
        ).generate()

        xy = base_mesh.node_coordinates.copy()
        x = xy[:, 0]
        y = xy[:, 1]

        eps = 1e-12

        dist_outer = np.minimum.reduce(
            [
                x - 0.0,
                g.outer_width - x,
                y - 0.0,
                g.outer_height - y,
            ]
        )
        dist_hole = _distance_to_axis_aligned_rectangle(
            x,
            y,
            x_min=g.hole_x_min,
            x_max=g.hole_x_max,
            y_min=g.hole_y_min,
            y_max=g.hole_y_max,
        )

        max_outer = float(np.max(dist_outer) + eps)
        max_hole = float(np.max(dist_hole) + eps)
        mask = (
            np.clip(dist_outer / max_outer, 0.0, 1.0) ** self.p
        ) * (
            np.clip(dist_hole / max_hole, 0.0, 1.0) ** self.p
        )

        def gaussian(xc: float, yc: float, sx: float, sy: float) -> np.ndarray:
            return np.exp(
                -0.5 * (((x - xc) / sx) ** 2 + ((y - yc) / sy) ** 2)
            )

        x_mid_hole = 0.5 * (g.hole_x_min + g.hole_x_max)
        y_mid_hole = 0.5 * (g.hole_y_min + g.hole_y_max)

        right_margin = g.outer_width - g.hole_x_max
        top_margin = g.outer_height - g.hole_y_max
        bottom_margin = g.hole_y_min
        left_margin = g.hole_x_min

        x_a = g.hole_x_max
        y_a = g.hole_y_min

        load_w = gaussian(
            x_mid_hole,
            g.hole_y_max + 0.30 * top_margin,
            sx=0.18 * g.outer_width,
            sy=0.10 * g.outer_height,
        )

        right_leg_w = gaussian(
            g.hole_x_max + 0.30 * right_margin,
            y_mid_hole,
            sx=0.10 * g.outer_width,
            sy=0.22 * g.outer_height,
        )

        a_w = gaussian(
            x_a + 0.28 * right_margin,
            y_a + 0.28 * bottom_margin,
            sx=0.10 * g.outer_width,
            sy=0.10 * g.outer_height,
        )

        inner_left_corner_w = gaussian(
            g.hole_x_min - 0.22 * left_margin,
            g.hole_y_max + 0.22 * top_margin,
            sx=0.10 * g.outer_width,
            sy=0.10 * g.outer_height,
        )

        inner_right_corner_w = gaussian(
            g.hole_x_max + 0.22 * right_margin,
            g.hole_y_max + 0.22 * top_margin,
            sx=0.10 * g.outer_width,
            sy=0.10 * g.outer_height,
        )

        bl_w = gaussian(
            0.12 * g.outer_width,
            0.12 * g.outer_height,
            sx=0.16 * g.outer_width,
            sy=0.16 * g.outer_height,
        )

        tr_w = gaussian(
            0.88 * g.outer_width,
            0.88 * g.outer_height,
            sx=0.16 * g.outer_width,
            sy=0.16 * g.outer_height,
        )

        field_w = np.clip(
            1.10 * load_w
            + 1.00 * right_leg_w
            + 0.95 * a_w
            + 0.75 * inner_left_corner_w
            + 0.75 * inner_right_corner_w
            + 0.70 * bl_w
            + 0.70 * tr_w,
            0.0,
            None,
        )

        u = (y - 0.5 * g.outer_height) / max(g.outer_height, eps)
        v = (x - 0.5 * g.outer_width) / max(g.outer_width, eps)

        x_warp = x + self.amplitude * mask * field_w * (0.90 * u + 0.20 * v)
        y_warp = y - 0.75 * self.amplitude * mask * field_w * (0.90 * v - 0.15 * u)

        warped_xy = np.column_stack([x_warp, y_warp])

        return HeterosisMesh.from_arrays(
            node_coordinates=warped_xy,
            w_location_matrix=base_mesh.w_location_matrix,
        )