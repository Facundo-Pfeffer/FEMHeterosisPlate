from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plate_fea.elements import HeterosisPlateElement
from plate_fea.mesh import HeterosisMesh
from plate_fea.mesh_generation import generate_rectangular_heterosis_mesh
from plate_fea.model import PlateModel
from plate_fea.quadrature import tensor_product_rule

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output" / "integration_patch_diagnostics"


@dataclass(frozen=True)
class RepresentableDistortedPatchKinematicField:
    """
    Coefficients for a distorted-patch kinematic field representable by the
    standard heterosis interpolation.

    The prescribed transverse displacement field is linear in physical coordinates:

        w(x, y) = w_x_slope * x + w_y_slope * y + w_offset

    The prescribed rotation fields are linear in physical coordinates:

        theta_x(x, y) =
            curvature_xx * x
            + 0.5 * curvature_xy * y
            + theta_x_intercept

        theta_y(x, y) =
            0.5 * curvature_xy * x
            + curvature_yy * y
            + theta_y_intercept

    With the shear convention used in these tests,

        gamma_xz = w_,x - theta_x
        gamma_yz = w_,y - theta_y

    the bending strains are constant, while the shear strains are known functions
    of x and y.
    """

    curvature_xx: float = 0.13
    curvature_yy: float = 0.17
    curvature_xy: float = -0.42

    theta_x_intercept: float = 0.08
    theta_y_intercept: float = -0.04

    w_x_slope: float = 0.19
    w_y_slope: float = -0.11
    w_offset: float = 0.30


def build_distorted_five_element_patch() -> HeterosisMesh:
    """
    Build a 5-element patch whose overall boundary is rectangular.

    The topology is: top/left/right/bottom elements enclosing one distorted
    middle element.
    """
    node_coordinates = np.array(
        [
            [0.0, 0.0],  # 0  outer bottom-left
            [3.4, 0.0],  # 1  outer bottom-right
            [3.4, 3.0],  # 2  outer top-right
            [0.0, 3.0],  # 3  outer top-left
            [1.15, 1.05],  # 4  inner bottom-left
            [2.28, 1.00],  # 5  inner bottom-right
            [2.20, 2.08],  # 6  inner top-right
            [1.02, 2.15],  # 7  inner top-left
            [1.70, 0.0],  # 8  outer bottom midside
            [3.4, 1.5],  # 9  outer right midside
            [1.70, 3.0],  # 10 outer top midside
            [0.0, 1.5],  # 11 outer left midside
            [1.715, 1.025],  # 12 inner bottom midside
            [2.24, 1.54],  # 13 inner right midside
            [1.61, 2.115],  # 14 inner top midside
            [1.085, 1.60],  # 15 inner left midside
            [2.80, 2.54],  # 16 connector midside: outer top-right <-> inner top-right
            [0.51, 2.575],  # 17 connector midside: outer top-left <-> inner top-left
            [2.84, 0.50],  # 18 connector midside: outer bottom-right <-> inner bottom-right
            [0.575, 0.525],  # 19 connector midside: outer bottom-left <-> inner bottom-left
        ],
        dtype=float,
    )

    # Column order: top, left, center, right, bottom.
    w_location_matrix = np.array(
        [
            [3, 0, 4, 2, 0],  # local corner 0
            [2, 3, 5, 1, 1],  # local corner 1
            [6, 7, 6, 5, 5],  # local corner 2
            [7, 4, 7, 6, 4],  # local corner 3
            [10, 11, 12, 9, 8],  # local midside 0-1
            [16, 17, 13, 18, 18],  # local midside 1-2
            [14, 15, 14, 13, 12],  # local midside 2-3
            [17, 19, 15, 16, 19],  # local midside 3-0
        ],
        dtype=int,
    )

    return HeterosisMesh.from_arrays(
        node_coordinates=node_coordinates,
        w_location_matrix=w_location_matrix,
    )


def build_distorted_single_element_mesh() -> HeterosisMesh:
    base_mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=1, ny=1)

    node_coordinates = base_mesh.node_coordinates.copy()
    node_coordinates[2] += np.array([0.08, -0.05])
    node_coordinates[4] += np.array([0.06, 0.00])
    node_coordinates[5] += np.array([0.00, -0.07])
    node_coordinates[6] += np.array([-0.05, 0.00])
    node_coordinates[7] += np.array([0.00, 0.06])

    return HeterosisMesh.from_arrays(
        node_coordinates=node_coordinates,
        w_location_matrix=base_mesh.w_location_matrix,
    )


def assemble_representable_distorted_patch_displacement_vector(
    mesh: HeterosisMesh,
    model: PlateModel,
    prescribed_kinematic_field: RepresentableDistortedPatchKinematicField = RepresentableDistortedPatchKinematicField(),
) -> tuple[np.ndarray, RepresentableDistortedPatchKinematicField]:
    """
    Build the global displacement vector for the representable distorted-patch field.

    Returns:
        global_displacement_vector:
            One-dimensional array containing all global model degrees of freedom.

        prescribed_kinematic_field:
            Coefficients used to define the analytical w, theta_x, and theta_y fields.
    """
    global_displacement_vector = np.zeros(mesh.total_dof_number, dtype=float)

    w_node_coordinates = mesh.node_coordinates
    w_x_coordinates = w_node_coordinates[:, 0]
    w_y_coordinates = w_node_coordinates[:, 1]

    global_displacement_vector[: mesh.total_w_node_number] = (
        prescribed_kinematic_field.w_x_slope * w_x_coordinates
        + prescribed_kinematic_field.w_y_slope * w_y_coordinates
        + prescribed_kinematic_field.w_offset
    )

    theta_node_coordinates = mesh.theta_node_coordinates
    theta_x_coordinates = theta_node_coordinates[:, 0]
    theta_y_coordinates = theta_node_coordinates[:, 1]

    theta_x_values = (
        prescribed_kinematic_field.curvature_xx * theta_x_coordinates
        + 0.5 * prescribed_kinematic_field.curvature_xy * theta_y_coordinates
        + prescribed_kinematic_field.theta_x_intercept
    )

    theta_y_values = (
        0.5 * prescribed_kinematic_field.curvature_xy * theta_x_coordinates
        + prescribed_kinematic_field.curvature_yy * theta_y_coordinates
        + prescribed_kinematic_field.theta_y_intercept
    )

    for theta_node_id in range(mesh.total_theta_node_number):
        global_displacement_vector[model.get_theta_x_dof(theta_node_id)] = float(theta_x_values[theta_node_id])
        global_displacement_vector[model.get_theta_y_dof(theta_node_id)] = float(theta_y_values[theta_node_id])

    return global_displacement_vector, prescribed_kinematic_field


def evaluate_expected_distorted_patch_strains_at_sample_points(
    sampled_strain_fields: dict[str, np.ndarray],
    prescribed_kinematic_field: RepresentableDistortedPatchKinematicField,
) -> dict[str, np.ndarray]:
    """
    Evaluate the analytical strain fields at the sampled quadrature-point coordinates.

    The sampled_strain_fields dictionary must contain:
        - "x": physical x-coordinates of sampled quadrature points.
        - "y": physical y-coordinates of sampled quadrature points.
    """
    sample_x_coordinates = sampled_strain_fields["x"]
    sample_y_coordinates = sampled_strain_fields["y"]

    theta_x_at_sample_points = (
        prescribed_kinematic_field.curvature_xx * sample_x_coordinates
        + 0.5 * prescribed_kinematic_field.curvature_xy * sample_y_coordinates
        + prescribed_kinematic_field.theta_x_intercept
    )

    theta_y_at_sample_points = (
        0.5 * prescribed_kinematic_field.curvature_xy * sample_x_coordinates
        + prescribed_kinematic_field.curvature_yy * sample_y_coordinates
        + prescribed_kinematic_field.theta_y_intercept
    )

    return {
        "kappa_xx": np.full_like(sample_x_coordinates, prescribed_kinematic_field.curvature_xx),
        "kappa_yy": np.full_like(sample_x_coordinates, prescribed_kinematic_field.curvature_yy),
        "kappa_xy": np.full_like(sample_x_coordinates, prescribed_kinematic_field.curvature_xy),
        "gamma_xz": prescribed_kinematic_field.w_x_slope - theta_x_at_sample_points,
        "gamma_yz": prescribed_kinematic_field.w_y_slope - theta_y_at_sample_points,
    }


def compute_generalized_strains_at_quadrature_points(
    mesh: HeterosisMesh,
    model: PlateModel,
    global_displacement_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Evaluate generalized strain fields at all 3x3 quadrature points of every element.

    Returns:
        Dictionary containing:
            - "x", "y": physical coordinates of each sampled point.
            - "kappa_xx", "kappa_yy", "kappa_xy": bending strain samples.
            - "gamma_xz", "gamma_yz": shear strain samples.
    """
    element = HeterosisPlateElement()
    quadrature_points = tensor_product_rule(order_x=3, order_y=3).points

    sampled_fields: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "kappa_xx": [],
        "kappa_yy": [],
        "kappa_xy": [],
        "gamma_xz": [],
        "gamma_yz": [],
    }

    for element_id in range(mesh.total_element_number):
        element_geometry_coordinates = mesh.get_geometry_coordinates(element_id)
        element_global_dof_indices = element.local_to_global_dof_indices(mesh, element_id)
        element_displacement_vector = global_displacement_vector[element_global_dof_indices]

        for parent_point in quadrature_points:
            xi = float(parent_point[0])
            eta = float(parent_point[1])

            geometry_jacobian = element.geometry_jacobian(
                xi,
                eta,
                element_geometry_coordinates,
            )

            q8_shape_values = element.q8_shape_functions(xi, eta)
            x_sample_coordinate = float(q8_shape_values @ element_geometry_coordinates[:, 0])
            y_sample_coordinate = float(q8_shape_values @ element_geometry_coordinates[:, 1])

            sampled_fields["x"].append(x_sample_coordinate)
            sampled_fields["y"].append(y_sample_coordinate)

            d_q9_dxi, d_q9_deta = element.q9_shape_function_gradients_parent(xi, eta)
            d_q9_dx, d_q9_dy = element.parent_to_physical_gradients(
                d_q9_dxi,
                d_q9_deta,
                geometry_jacobian,
            )
            bending_strain = element.bending_B_matrix(d_q9_dx, d_q9_dy) @ element_displacement_vector

            d_q8_dxi, d_q8_deta = element.q8_shape_function_gradients_parent(xi, eta)
            d_q8_dx, d_q8_dy = element.parent_to_physical_gradients(
                d_q8_dxi,
                d_q8_deta,
                geometry_jacobian,
            )
            q9_shape_values = element.q9_shape_functions(xi, eta)
            shear_strain = element.shear_B_matrix(
                d_q8_dx,
                d_q8_dy,
                q9_shape_values,
            ) @ element_displacement_vector

            sampled_fields["kappa_xx"].append(float(bending_strain[0]))
            sampled_fields["kappa_yy"].append(float(bending_strain[1]))
            sampled_fields["kappa_xy"].append(float(bending_strain[2]))
            sampled_fields["gamma_xz"].append(float(shear_strain[0]))
            sampled_fields["gamma_yz"].append(float(shear_strain[1]))

    return {
        field_name: np.asarray(field_values, dtype=float)
        for field_name, field_values in sampled_fields.items()
    }


def plot_patch_geometry(mesh: HeterosisMesh, outdir: Path) -> Path:
    path = outdir / "patch_geometry_5element_distorted.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    node_coordinates = mesh.node_coordinates
    w_location_matrix = mesh.w_location_matrix

    edges = ((0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0))
    labels = {
        0: r"$E_{\mathrm{top}}$",
        1: r"$E_{\mathrm{left}}$",
        2: r"$E_{\mathrm{center}}$",
        3: r"$E_{\mathrm{right}}$",
        4: r"$E_{\mathrm{bottom}}$",
    }
    label_offsets_by_element = {
        0: (0.00, 0.18),
        1: (-0.22, 0.00),
        2: (-0.10, -0.10),
        3: (0.22, 0.00),
        4: (0.00, -0.18),
    }

    for element_id in range(w_location_matrix.shape[1]):
        element_node_ids = w_location_matrix[:, element_id]
        element_node_coordinates = node_coordinates[element_node_ids]

        for i, j in edges:
            ax.plot(
                [element_node_coordinates[i, 0], element_node_coordinates[j, 0]],
                [element_node_coordinates[i, 1], element_node_coordinates[j, 1]],
                "k-",
                lw=1.4,
            )

        center_x = float(np.mean(element_node_coordinates[:, 0]))
        center_y = float(np.mean(element_node_coordinates[:, 1]))
        offset_x, offset_y = label_offsets_by_element[element_id]

        ax.text(
            center_x + offset_x,
            center_y + offset_y,
            labels[element_id],
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )

    ax.scatter(
        node_coordinates[:, 0],
        node_coordinates[:, 1],
        s=20,
        c="tab:red",
        zorder=3,
        label="Q8 and Q9 element nodes",
    )

    q9_center_theta_ids = np.unique(mesh.theta_location_matrix[8, :])
    q9_center_coordinates = mesh.theta_node_coordinates[q9_center_theta_ids, :]

    ax.scatter(
        q9_center_coordinates[:, 0],
        q9_center_coordinates[:, 1],
        s=20,
        c="tab:blue",
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="Q9 center nodes",
    )

    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("y (unitless)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)

    handles, label_text = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        label_text,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )

    fig.suptitle("Distorted 5-element patch (enclosing topology)", y=0.98)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.88])
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_field_maps(
    sampled_strain_fields: dict[str, np.ndarray],
    expected_strain_fields: dict[str, np.ndarray],
    outdir: Path,
) -> Path:
    path = outdir / "strain_fields_component_maps.png"

    x_coordinates = sampled_strain_fields["x"]
    y_coordinates = sampled_strain_fields["y"]
    strain_field_names = ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs_flat = axs.flatten()

    for field_index, strain_field_name in enumerate(strain_field_names):
        ax = axs_flat[field_index]
        sampled_values = sampled_strain_fields[strain_field_name]
        expected_values = expected_strain_fields[strain_field_name]

        contour = ax.tricontourf(
            x_coordinates,
            y_coordinates,
            sampled_values,
            levels=18,
            cmap="viridis",
        )
        fig.colorbar(contour, ax=ax, shrink=0.85)

        spread = float(np.max(sampled_values) - np.min(sampled_values))
        max_abs_error = float(np.max(np.abs(sampled_values - expected_values)))

        ax.set_title(f"{strain_field_name}\nspread={spread:.3e}, max|err|={max_abs_error:.3e}")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.15)

    axs_flat[-1].axis("off")
    axs_flat[-1].text(
        0.0,
        0.95,
        "How fields are accessed:\n"
        "1) local_dofs = element.local_to_global_dof_indices(mesh, eid)\n"
        "2) u_local = u[local_dofs]\n"
        "3) kappa = B_b(xi,eta) @ u_local\n"
        "4) gamma = B_s(xi,eta) @ u_local\n"
        "5) Repeat at all quadrature points per element",
        va="top",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle("Five strain fields sampled component-wise on distorted patch", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_eigenvalues(eigenvalues: np.ndarray, near_zero_count: int, outdir: Path) -> Path:
    path = outdir / "distorted_single_element_eigenvalues.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    eigenvalue_indices = np.arange(1, eigenvalues.size + 1)
    clipped_eigenvalues = np.clip(eigenvalues, 1.0e-18, None)

    ax.semilogy(eigenvalue_indices, clipped_eigenvalues, "o-", lw=1.3, ms=4)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title(f"Unsupported distorted element spectrum (near-zero count: {near_zero_count})")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def plot_single_element_geometry(mesh: HeterosisMesh, outdir: Path) -> Path:
    path = outdir / "distorted_single_element_geometry.png"

    fig, ax = plt.subplots(figsize=(6, 6))
    node_coordinates = mesh.node_coordinates
    element_node_ids = mesh.w_location_matrix[:, 0]
    element_node_coordinates = node_coordinates[element_node_ids]
    edges = ((0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0))

    for i, j in edges:
        ax.plot(
            [element_node_coordinates[i, 0], element_node_coordinates[j, 0]],
            [element_node_coordinates[i, 1], element_node_coordinates[j, 1]],
            "k-",
            lw=1.4,
        )

    center_x = float(np.mean(element_node_coordinates[:, 0]))
    center_y = float(np.mean(element_node_coordinates[:, 1]))

    ax.text(
        center_x - 0.08,
        center_y - 0.08,
        r"$E_{\mathrm{single}}$",
        fontsize=10,
        ha="center",
        va="center",
        color="black",
    )

    ax.scatter(
        node_coordinates[:, 0],
        node_coordinates[:, 1],
        s=20,
        c="tab:red",
        zorder=3,
        label="Q8 and Q9 element nodes",
    )

    q9_center_theta_ids = np.unique(mesh.theta_location_matrix[8, :])
    q9_center_coordinates = mesh.theta_node_coordinates[q9_center_theta_ids, :]

    ax.scatter(
        q9_center_coordinates[:, 0],
        q9_center_coordinates[:, 1],
        s=20,
        c="tab:blue",
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="Q9 center nodes",
    )

    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("y (unitless)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)

    handles, label_text = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        label_text,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )

    fig.suptitle("Distorted single-element geometry", y=0.98)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.88])
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path