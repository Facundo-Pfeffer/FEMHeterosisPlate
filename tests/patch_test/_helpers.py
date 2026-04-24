from __future__ import annotations

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


def build_distorted_five_element_patch() -> HeterosisMesh:
    """
    Build a 5-element patch whose overall boundary is rectangular.

    The topology is: top/left/right/bottom elements enclosing one distorted middle element.
    """
    node_coordinates = np.array(
        [
            [0.0, 0.0],  # 0  outer bottom-left
            [3.4, 0.0],  # 1  outer bottom-right
            [3.4, 3.0],  # 2  outer top-right
            [0.0, 3.0],  # 3  outer top-left
            [1.15, 1.05],  # 4  inner bottom-left (distorted center element corner)
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

    return HeterosisMesh.from_arrays(node_coordinates=node_coordinates, w_location_matrix=w_location_matrix)


def build_distorted_single_element_mesh() -> HeterosisMesh:
    base_mesh = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=1, ny=1)
    node_coordinates = base_mesh.node_coordinates.copy()
    node_coordinates[2] += np.array([0.08, -0.05])
    node_coordinates[4] += np.array([0.06, 0.00])
    node_coordinates[5] += np.array([0.00, -0.07])
    node_coordinates[6] += np.array([-0.05, 0.00])
    node_coordinates[7] += np.array([0.00, 0.06])
    return HeterosisMesh.from_arrays(node_coordinates=node_coordinates, w_location_matrix=base_mesh.w_location_matrix)


def assemble_polynomial_state(mesh: HeterosisMesh, model: PlateModel) -> tuple[np.ndarray, dict[str, float]]:
    a, b, e = 0.13, -0.21, 0.17
    c, f = 0.08, -0.04
    gx, gy = 0.19, -0.11
    k0 = 0.3

    global_displacement = np.zeros(mesh.total_dof_number, dtype=float)
    total_w_nodes = mesh.total_w_node_number

    w_node_coordinates = mesh.node_coordinates
    w_x_coordinates = w_node_coordinates[:, 0]
    w_y_coordinates = w_node_coordinates[:, 1]
    global_displacement[:total_w_nodes] = (
        0.5 * a * w_x_coordinates * w_x_coordinates
        + b * w_x_coordinates * w_y_coordinates
        + 0.5 * e * w_y_coordinates * w_y_coordinates
        + (c + gx) * w_x_coordinates
        + (f + gy) * w_y_coordinates
        + k0
    )

    theta_node_coordinates = mesh.theta_node_coordinates
    theta_x_coordinates = theta_node_coordinates[:, 0]
    theta_y_coordinates = theta_node_coordinates[:, 1]
    theta_x_values = a * theta_x_coordinates + b * theta_y_coordinates + c
    theta_y_values = b * theta_x_coordinates + e * theta_y_coordinates + f
    for theta_node_id in range(mesh.total_theta_node_number):
        global_displacement[model.get_theta_x_dof(theta_node_id)] = float(theta_x_values[theta_node_id])
        global_displacement[model.get_theta_y_dof(theta_node_id)] = float(theta_y_values[theta_node_id])

    expected_generalized_strains = {
        "kappa_xx": a,
        "kappa_yy": e,
        "kappa_xy": 2.0 * b,
        "gamma_xz": gx,
        "gamma_yz": gy,
    }
    return global_displacement, expected_generalized_strains


def compute_generalized_strains_at_quadrature_points(
    mesh: HeterosisMesh, model: PlateModel, u: np.ndarray
) -> dict[str, np.ndarray]:
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
        element_displacement = u[element_global_dof_indices]
        for parent_point in quadrature_points:
            xi = float(parent_point[0])
            eta = float(parent_point[1])
            geometry_jacobian = element.geometry_jacobian(xi, eta, element_geometry_coordinates)

            q8_shape_values = element.q8_shape_functions(xi, eta)
            x_quadrature = float(q8_shape_values @ element_geometry_coordinates[:, 0])
            y_quadrature = float(q8_shape_values @ element_geometry_coordinates[:, 1])
            sampled_fields["x"].append(x_quadrature)
            sampled_fields["y"].append(y_quadrature)

            d_q9_dxi, d_q9_deta = element.q9_shape_function_gradients_parent(xi, eta)
            d_q9_dx, d_q9_dy = element.parent_to_physical_gradients(d_q9_dxi, d_q9_deta, geometry_jacobian)
            bending_strain = element.bending_B_matrix(d_q9_dx, d_q9_dy) @ element_displacement

            d_q8_dxi, d_q8_deta = element.q8_shape_function_gradients_parent(xi, eta)
            d_q8_dx, d_q8_dy = element.parent_to_physical_gradients(d_q8_dxi, d_q8_deta, geometry_jacobian)
            q9_shape_values = element.q9_shape_functions(xi, eta)
            shear_strain = element.shear_B_matrix(d_q8_dx, d_q8_dy, q9_shape_values) @ element_displacement

            sampled_fields["kappa_xx"].append(float(bending_strain[0]))
            sampled_fields["kappa_yy"].append(float(bending_strain[1]))
            sampled_fields["kappa_xy"].append(float(bending_strain[2]))
            sampled_fields["gamma_xz"].append(float(shear_strain[0]))
            sampled_fields["gamma_yz"].append(float(shear_strain[1]))

    return {field_name: np.asarray(field_values, dtype=float) for field_name, field_values in sampled_fields.items()}


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
    # Keep semantic labels near their element centers with explicit per-element offsets.
    label_offsets_by_element = {
        0: (0.00, 0.18),  # E_top
        1: (-0.22, 0.00),  # E_left
        2: (-0.10, -0.10),  # E_center (away from center Q9 marker)
        3: (0.22, 0.00),  # E_right
        4: (0.00, -0.18),  # E_bottom
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
        cx = float(np.mean(element_node_coordinates[:, 0]))
        cy = float(np.mean(element_node_coordinates[:, 1]))
        dx, dy = label_offsets_by_element[element_id]
        ax.text(
            cx + dx,
            cy + dy,
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
    q9_center_xy = mesh.theta_node_coordinates[q9_center_theta_ids, :]
    ax.scatter(
        q9_center_xy[:, 0],
        q9_center_xy[:, 1],
        s=20,
        c="tab:blue",
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="Q9 center nodes.",
    )
    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("y (unitless)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)
    # Use a figure-level legend strip below the title (outside axes) to prevent overlaps.
    handles, labels_text = ax.get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=True)
    fig.suptitle("Distorted 5-element patch (enclosing topology)", y=0.98)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.88])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_field_maps(samples: dict[str, np.ndarray], expected: dict[str, float], outdir: Path) -> Path:
    path = outdir / "strain_fields_component_maps.png"
    x_coordinates = samples["x"]
    y_coordinates = samples["y"]
    fields = ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs_flat = axs.flatten()
    for field_index, field_name in enumerate(fields):
        ax = axs_flat[field_index]
        field_values = samples[field_name]
        contour = ax.tricontourf(x_coordinates, y_coordinates, field_values, levels=18, cmap="viridis")
        fig.colorbar(contour, ax=ax, shrink=0.85)
        spread = float(np.max(field_values) - np.min(field_values))
        max_abs_error = float(np.max(np.abs(field_values - expected[field_name])))
        ax.set_title(f"{field_name}\nspread={spread:.3e}, max|err|={max_abs_error:.3e}")
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
    """Plot the distorted single element with the same node style as patch geometry."""
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

    cx = float(np.mean(element_node_coordinates[:, 0]))
    cy = float(np.mean(element_node_coordinates[:, 1]))
    ax.text(cx - 0.08, cy - 0.08, r"$E_{\mathrm{single}}$", fontsize=10, ha="center", va="center", color="black")

    ax.scatter(
        node_coordinates[:, 0],
        node_coordinates[:, 1],
        s=20,
        c="tab:red",
        zorder=3,
        label="Q8 and Q9 element nodes",
    )
    q9_center_theta_ids = np.unique(mesh.theta_location_matrix[8, :])
    q9_center_xy = mesh.theta_node_coordinates[q9_center_theta_ids, :]
    ax.scatter(
        q9_center_xy[:, 0],
        q9_center_xy[:, 1],
        s=20,
        c="tab:blue",
        marker="o",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label="Q9 center nodes.",
    )

    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("y (unitless)")
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)
    handles, labels_text = ax.get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=True)
    fig.suptitle("Distorted single-element geometry", y=0.98)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.88])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
