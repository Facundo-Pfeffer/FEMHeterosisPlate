"""
Mesh generators for the Heterosis plate finite element.

Module layout
-------------
1. PlateWithHoleGeometry
       Frozen dataclass that describes the plate-with-hole geometry.
       Shared by all plate-with-hole generators and by ProblemConfig.

2. MeshGenerator
       Protocol satisfied by any mesh generator object.

3. Private helpers  (_node_key, _build_q8_mesh_from_cartesian_lines)
       Internal building blocks consumed only by the generators below.

4. Simple geometry generators  (patch tests, unit tests, benchmarks)
       generate_rectangular_heterosis_mesh
       generate_centered_quarter_disk_heterosis_mesh

5. Plate-with-hole generators  (the assignment problem)
       UniformBufferRingQ8Generator        — structured mesh, uniform buffer
                                             spacing; recommended default.
       GmshBoundarySensitiveQ8Generator    — unstructured mesh, distance-field
                                             sizing; requires the gmsh package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plate_fea.mesh import HeterosisMesh


# ── Geometry dataclass ──────────────────────────────────────────────────────────

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


# ── MeshGenerator Protocol ───────────────────────────────────────────────────────

class MeshGenerator(Protocol):
    def generate(self) -> HeterosisMesh: ...


# ── Private helpers ──────────────────────────────────────────────────────────────

def _node_key(x: float, y: float, digits: int = 9) -> tuple[float, float]:
    return (round(x, digits), round(y, digits))


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


# ── Simple geometry generators (mostly used for testing) ────────────────────────

def generate_rectangular_heterosis_mesh(width: float, height: float, nx: int, ny: int) -> HeterosisMesh:
    """
    Generate a structured Q8 heterosis mesh that covers a full rectangle [0, width] × [0, height].

    Args:
        width:  Horizontal extent of the rectangle. Must be positive.
        height: Vertical extent of the rectangle. Must be positive.
        nx:     Number of elements along the x-axis. Must be >= 1.
        ny:     Number of elements along the y-axis. Must be >= 1.

    Returns:
        HeterosisMesh with nx × ny serendipity Q8 elements and the
        rotational (Heterosis) node added internally by HeterosisMesh.
    """
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


def generate_centered_quarter_disk_heterosis_mesh(
        radius: float,
        n_el: int,
) -> HeterosisMesh:
    """
    Generate a structured Q8 mesh for the quarter disk  x ≥ 0, y ≥ 0.

    An n_el × n_el element grid on the computational square [0, 1]² is mapped
    smoothly onto the quarter disk via the Fong (2011) mapping:

        x(s, t) = radius · s · √(1 − t²/2)
        y(s, t) = radius · t · √(1 − s²/2)

    This mapping places a non-degenerate element at the origin, so the w-node
    at r = 0 is well-defined. That property is required for concentrated-load
    circular-plate problems where the center deflection is the output of interest.

    Args:
        radius: Radius of the quarter disk. Must be positive.
        n_el:   Number of elements along each axis of the computational square.
                Must be >= 1.

    Returns:
        HeterosisMesh with n_el² serendipity Q8 elements mapped onto the
        quarter disk, plus the rotational node added internally by HeterosisMesh.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    if n_el < 1:
        raise ValueError("n_el must be >= 1.")

    node_id_by_grid_point: dict[tuple[int, int], int] = {}
    node_coordinates: list[tuple[float, float]] = []

    def map_to_quarter_disk(s: float, t: float) -> tuple[float, float]:
        x = radius * s * np.sqrt(1.0 - 0.5 * t ** 2)
        y = radius * t * np.sqrt(1.0 - 0.5 * s ** 2)
        return float(x), float(y)

    def get_node_id(i_half: int, j_half: int) -> int:
        key = (i_half, j_half)

        if key in node_id_by_grid_point:
            return node_id_by_grid_point[key]

        s = i_half / (2.0 * n_el)
        t = j_half / (2.0 * n_el)

        node_id = len(node_coordinates)
        node_id_by_grid_point[key] = node_id
        node_coordinates.append(map_to_quarter_disk(s, t))

        return node_id

    connectivity: list[list[int]] = []

    for j in range(n_el):
        for i in range(n_el):
            n1 = get_node_id(2 * i, 2 * j)
            n2 = get_node_id(2 * i + 2, 2 * j)
            n3 = get_node_id(2 * i + 2, 2 * j + 2)
            n4 = get_node_id(2 * i, 2 * j + 2)
            n5 = get_node_id(2 * i + 1, 2 * j)
            n6 = get_node_id(2 * i + 2, 2 * j + 1)
            n7 = get_node_id(2 * i + 1, 2 * j + 2)
            n8 = get_node_id(2 * i, 2 * j + 1)

            connectivity.append([n1, n2, n3, n4, n5, n6, n7, n8])

    return HeterosisMesh.from_arrays(
        node_coordinates=np.asarray(node_coordinates, dtype=float),
        w_location_matrix=np.asarray(connectivity, dtype=int).T,
    )


# ── Plate-with-hole generators ───────────────────────────────────────────────────

@dataclass(frozen=True)
class UniformBufferRingQ8Generator:
    """
    Structured Q8 heterosis mesh for a rectangular plate with a centered rectangular hole.

    The domain is divided into five x-bands and five y-bands, producing a
    cross-shaped exclusion zone over the hole. A uniform buffer band of
    thickness ``buffer`` separates the hole boundary from the outer mesh region,
    concentrating elements near the hole without requiring manual zone tuning.

    Args:
        geometry:    Plate and hole dimensions. Defaults to PlateWithHoleGeometry().
        resolution:  Base element count per zone segment; larger values produce
                     a finer mesh everywhere. Must be >= -1.
        hole_refine: Extra elements added to the two hole-adjacent buffer bands.
                     Must be >= 0.
        buffer:      Thickness of the uniform buffer band surrounding the hole
                     boundary on all four sides. Must be > 0 and small enough
                     that the buffer rectangle fits strictly between the hole
                     and the outer plate boundary.
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
class GmshBoundarySensitiveQ8Generator:
    """
    Gmsh-based unstructured Q8 mesh with automatic distance-field element sizing.

    Gmsh places finer elements near the loaded hole-top edge and near the
    clamped/free BC-transition corners (bottom-left and top-right of the outer
    boundary), then coarsens toward the interior. This yields better element
    quality near stress concentrations than the structured generator without
    requiring manual zone tuning.

    Requires the gmsh Python package (``pip install --upgrade gmsh``) and the
    system library ``libGLU.so.1`` (``apt install libglu1-mesa`` on Debian/Ubuntu).

    Args:
        geometry:            Plate and hole dimensions. Defaults to PlateWithHoleGeometry().
        resolution:          Global density level; larger values produce a finer mesh
                             everywhere. Must be >= -1.
        hole_refine:         Extra refinement applied near the hole boundary and near
                             BC-transition corners. Must be >= 0.
        clamped_outer_edges: Names of the outer edges that carry a clamped boundary
                             condition. Used only to identify the clamped/free transition
                             corners that receive additional refinement.
    """

    geometry: PlateWithHoleGeometry = PlateWithHoleGeometry()
    resolution: int = 2
    hole_refine: int = 2
    clamped_outer_edges: tuple[str, ...] = ("left", "top")

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
