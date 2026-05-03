from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plate_fea.boundary_conditions import EssentialBoundaryCondition, NodalPointLoad
from plate_fea.elements import HeterosisPlateElement
from plate_fea.materials import PlateMaterial
from plate_fea.mesh_generation import generate_centered_quarter_disk_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.postprocessing import sample_w_at_quadrature_points
from plate_fea.solver import solve_displacement_system


E = 10.92e5
NU = 0.3
R = 5.0
T = 2.0
P = 1.0
KAPPA = 5.0 / 6.0
N_EL = 12

TOL = 1.0e-9
MAX_RELATIVE_L2_ERROR = 0.25

OUTPUT_DIR = Path("output/hughes")
T_LABEL = f"{T:g}".replace(".", "_")
PLOT_FILE = OUTPUT_DIR / f"hughes_cl_c_heterosis_t_{T_LABEL}.png"
CSV_FILE = OUTPUT_DIR / f"hughes_cl_c_heterosis_t_{T_LABEL}.csv"


def make_material() -> PlateMaterial:
    return PlateMaterial(
        young_modulus=E,
        poisson_ratio=NU,
        thickness=T,
        shear_correction_factor=KAPPA,
    )


def bending_rigidity(material: PlateMaterial) -> float:
    return float(material.bending_constitutive_matrix[0, 0])


def shear_rigidity(material: PlateMaterial) -> float:
    return float(material.shear_constitutive_matrix[0, 0])


def normalized_displacement(
    w: np.ndarray,
    material: PlateMaterial,
) -> np.ndarray:
    return 16.0 * np.pi * bending_rigidity(material) * w / (P * R**2)


def reissner_clamped_center_load_w(
    r: np.ndarray,
    material: PlateMaterial,
) -> np.ndarray:
    """
    Axisymmetric Reissner-Mindlin reference for a clamped circular plate
    under a centered concentrated load.

    The point-load shear correction is singular at r = 0, so the comparison
    is made at positive-radius quadrature points.
    """
    rho = np.asarray(r, dtype=float) / R
    rho = np.clip(rho, 1.0e-14, 1.0)

    d = bending_rigidity(material)
    kgt = shear_rigidity(material)

    w_kirchhoff = (
        P
        * R**2
        / (16.0 * np.pi * d)
        * (1.0 - rho**2 + 2.0 * rho**2 * np.log(rho))
    )

    w_shear = P / (2.0 * np.pi * kgt) * np.log(1.0 / rho)

    return w_kirchhoff + w_shear


def build_model() -> PlateModel:
    material = make_material()
    mesh = generate_centered_quarter_disk_heterosis_mesh(
        radius=R,
        n_el=N_EL,
    )

    return PlateModel(
        mesh=mesh,
        constitutive_material=material,
        element_formulation=HeterosisPlateElement(),
    )


def add_symmetry_boundary_conditions(model: PlateModel) -> None:
    """
    Symmetric loading on the two straight symmetry boundaries.

    With the repository convention

        gamma_xz = dw/dx - theta_x,
        gamma_yz = dw/dy - theta_y,

    symmetry gives:
        x = 0: theta_x = 0
        y = 0: theta_y = 0
    """
    theta_xy = model.mesh.theta_node_coordinates

    theta_nodes_on_x_equal_zero = np.flatnonzero(
        np.abs(theta_xy[:, 0]) < TOL
    )
    theta_nodes_on_y_equal_zero = np.flatnonzero(
        np.abs(theta_xy[:, 1]) < TOL
    )

    model.add_essential_condition(
        EssentialBoundaryCondition(
            "theta_x",
            theta_nodes_on_x_equal_zero.tolist(),
            0.0,
        )
    )
    model.add_essential_condition(
        EssentialBoundaryCondition(
            "theta_y",
            theta_nodes_on_y_equal_zero.tolist(),
            0.0,
        )
    )


def add_clamped_outer_boundary(model: PlateModel) -> None:
    w_xy = model.mesh.node_coordinates
    theta_xy = model.mesh.theta_node_coordinates

    w_radius = np.linalg.norm(w_xy, axis=1)
    theta_radius = np.linalg.norm(theta_xy, axis=1)

    w_nodes_on_outer_radius = np.flatnonzero(np.abs(w_radius - R) < 5.0e-8)
    theta_nodes_on_outer_radius = np.flatnonzero(
        np.abs(theta_radius - R) < 5.0e-8
    )

    model.add_essential_condition(
        EssentialBoundaryCondition(
            "w",
            w_nodes_on_outer_radius.tolist(),
            0.0,
        )
    )

    for field_name in ("theta_x", "theta_y"):
        model.add_essential_condition(
            EssentialBoundaryCondition(
                field_name,
                theta_nodes_on_outer_radius.tolist(),
                0.0,
            )
        )


def center_w_node_id(model: PlateModel) -> int:
    radius = np.linalg.norm(model.mesh.node_coordinates, axis=1)
    node_id = int(np.argmin(radius))

    if radius[node_id] > TOL:
        raise ValueError(
            "The mesh does not contain a true center w-node. "
            f"Closest radius is {radius[node_id]:.6e}."
        )

    return node_id


def add_center_point_load(model: PlateModel) -> None:
    center_node = center_w_node_id(model)

    model.add_nodal_load(
        NodalPointLoad(
            field_name="w",
            node_id=center_node,
            value=P / 4.0,
        )
    )


def solve_hughes_model() -> tuple[PlateModel, np.ndarray]:
    model = build_model()

    add_symmetry_boundary_conditions(model)
    add_clamped_outer_boundary(model)
    add_center_point_load(model)

    _, _, displacement = solve_displacement_system(model)

    return model, displacement


def save_profile_csv(
    rho: np.ndarray,
    y_fem: np.ndarray,
    y_ref: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.column_stack([rho, y_fem, y_ref])
    header = "r_over_R,normalized_w_fem,normalized_w_reissner"

    np.savetxt(
        CSV_FILE,
        data,
        delimiter=",",
        header=header,
        comments="",
    )


def plot_profile(
    rho_fem: np.ndarray,
    y_fem: np.ndarray,
    material: PlateMaterial,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rho_exact = np.linspace(1.0e-4, 1.0, 600)
    r_exact = rho_exact * R
    y_exact = normalized_displacement(
        reissner_clamped_center_load_w(r_exact, material),
        material,
    )

    order = np.argsort(rho_fem)

    fig, ax = plt.subplots(figsize=(5.0, 6.2))

    ax.plot(
        rho_exact,
        y_exact,
        linewidth=1.5,
        label="Exact solution (Reissner theory)",
    )
    ax.scatter(
        rho_fem[order],
        y_fem[order],
        marker="o",
        facecolors="none",
        edgecolors="black",
        label="Heterosis",
    )

    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$16\pi D w/(P R^2)$")

    ax.set_xlim(0.0, 1.0)

    y_max = max(float(np.nanmax(y_exact)), float(np.nanmax(y_fem)))
    ax.set_ylim(0.0, 1.08 * y_max)
    ax.invert_yaxis()

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_xticks(np.arange(0.0, 1.01, 0.05), minor=True)

    ax.grid(axis="x", which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    ax.grid(axis="x", which="minor", linestyle="--", linewidth=0.4, alpha=0.4)

    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=300)
    plt.close(fig)


def test_hughes_clamped_circular_plate_center_load_profile() -> None:
    model, displacement = solve_hughes_model()
    material = model.constitutive_material

    sampled_w = sample_w_at_quadrature_points(
        model.mesh,
        displacement,
        quadrature_order=(2, 2),
    )

    r_fem = np.hypot(sampled_w.x, sampled_w.y)
    rho_fem = r_fem / R

    y_fem = normalized_displacement(sampled_w.w, material)
    y_ref = normalized_displacement(
        reissner_clamped_center_load_w(r_fem, material),
        material,
    )

    plot_profile(rho_fem, y_fem, material)
    save_profile_csv(rho_fem, y_fem, y_ref)

    comparison_mask = (
        np.isfinite(y_fem)
        & np.isfinite(y_ref)
        & (rho_fem > 0.05)
        & (rho_fem < 0.95)
    )

    relative_l2_error = (
        np.linalg.norm(y_fem[comparison_mask] - y_ref[comparison_mask])
        / np.linalg.norm(y_ref[comparison_mask])
    )

    assert relative_l2_error < MAX_RELATIVE_L2_ERROR, (
        "Hughes clamped circular-plate center-load profile is outside tolerance. "
        f"relative_l2_error={relative_l2_error:.6f}, "
        f"limit={MAX_RELATIVE_L2_ERROR:.6f}, "
        f"plot={PLOT_FILE}, "
        f"csv={CSV_FILE}"
    )