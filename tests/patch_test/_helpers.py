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
    xy = np.array(
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
    wlm = np.array(
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

    return HeterosisMesh.from_arrays(node_coordinates=xy, w_location_matrix=wlm)


def build_distorted_single_element_mesh() -> HeterosisMesh:
    base = generate_rectangular_heterosis_mesh(width=1.0, height=1.0, nx=1, ny=1)
    xy = base.node_coordinates.copy()
    xy[2] += np.array([0.08, -0.05])
    xy[4] += np.array([0.06, 0.00])
    xy[5] += np.array([0.00, -0.07])
    xy[6] += np.array([-0.05, 0.00])
    xy[7] += np.array([0.00, 0.06])
    return HeterosisMesh.from_arrays(node_coordinates=xy, w_location_matrix=base.w_location_matrix)


def assemble_polynomial_state(mesh: HeterosisMesh, model: PlateModel) -> tuple[np.ndarray, dict[str, float]]:
    a, b, e = 0.13, -0.21, 0.17
    c, f = 0.08, -0.04
    gx, gy = 0.19, -0.11
    k0 = 0.3

    u = np.zeros(mesh.total_dof_number, dtype=float)
    n_w = mesh.total_w_node_number

    xy_w = mesh.node_coordinates
    xw = xy_w[:, 0]
    yw = xy_w[:, 1]
    u[:n_w] = 0.5 * a * xw * xw + b * xw * yw + 0.5 * e * yw * yw + (c + gx) * xw + (f + gy) * yw + k0

    xy_t = mesh.theta_node_coordinates
    xt = xy_t[:, 0]
    yt = xy_t[:, 1]
    tx = a * xt + b * yt + c
    ty = b * xt + e * yt + f
    for theta_node_id in range(mesh.total_theta_node_number):
        u[model.get_theta_x_dof(theta_node_id)] = float(tx[theta_node_id])
        u[model.get_theta_y_dof(theta_node_id)] = float(ty[theta_node_id])

    expected = {
        "kappa_xx": a,
        "kappa_yy": e,
        "kappa_xy": 2.0 * b,
        "gamma_xz": gx,
        "gamma_yz": gy,
    }
    return u, expected


def sample_strains(mesh: HeterosisMesh, model: PlateModel, u: np.ndarray) -> dict[str, np.ndarray]:
    element = HeterosisPlateElement()
    qp = tensor_product_rule(order_x=3, order_y=3).points

    out: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "kappa_xx": [],
        "kappa_yy": [],
        "kappa_xy": [],
        "gamma_xz": [],
        "gamma_yz": [],
    }
    for element_id in range(mesh.total_element_number):
        geom = mesh.get_geometry_coordinates(element_id)
        gdofs = element.local_to_global_dof_indices(mesh, element_id)
        u_local = u[gdofs]
        for xi_eta in qp:
            xi = float(xi_eta[0])
            eta = float(xi_eta[1])
            jac = element.geometry_jacobian(xi, eta, geom)

            n_w = element.q8_shape_functions(xi, eta)
            x_q = float(n_w @ geom[:, 0])
            y_q = float(n_w @ geom[:, 1])
            out["x"].append(x_q)
            out["y"].append(y_q)

            d_nt_dxi, d_nt_deta = element.q9_shape_function_gradients_parent(xi, eta)
            d_nt_dx, d_nt_dy = element.parent_to_physical_gradients(d_nt_dxi, d_nt_deta, jac)
            kappa = element.bending_B_matrix(d_nt_dx, d_nt_dy) @ u_local

            d_nw_dxi, d_nw_deta = element.q8_shape_function_gradients_parent(xi, eta)
            d_nw_dx, d_nw_dy = element.parent_to_physical_gradients(d_nw_dxi, d_nw_deta, jac)
            nt = element.q9_shape_functions(xi, eta)
            gamma = element.shear_B_matrix(d_nw_dx, d_nw_dy, nt) @ u_local

            out["kappa_xx"].append(float(kappa[0]))
            out["kappa_yy"].append(float(kappa[1]))
            out["kappa_xy"].append(float(kappa[2]))
            out["gamma_xz"].append(float(gamma[0]))
            out["gamma_yz"].append(float(gamma[1]))

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def plot_patch_geometry(mesh: HeterosisMesh, outdir: Path) -> Path:
    path = outdir / "patch_geometry_5element_distorted.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    xy = mesh.node_coordinates
    wlm = mesh.w_location_matrix
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
    for eid in range(wlm.shape[1]):
        ids = wlm[:, eid]
        exy = xy[ids]
        for i, j in edges:
            ax.plot([exy[i, 0], exy[j, 0]], [exy[i, 1], exy[j, 1]], "k-", lw=1.4)
        cx = float(np.mean(exy[:, 0]))
        cy = float(np.mean(exy[:, 1]))
        dx, dy = label_offsets_by_element[eid]
        ax.text(
            cx + dx,
            cy + dy,
            labels[eid],
            fontsize=10,
            ha="center",
            va="center",
            color="tab:blue",
        )
    ax.scatter(xy[:, 0], xy[:, 1], s=20, c="tab:red", zorder=3, label="Q8 and Q9 element nodes")
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
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_field_maps(samples: dict[str, np.ndarray], expected: dict[str, float], outdir: Path) -> Path:
    path = outdir / "strain_fields_component_maps.png"
    x = samples["x"]
    y = samples["y"]
    fields = ("kappa_xx", "kappa_yy", "kappa_xy", "gamma_xz", "gamma_yz")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs_flat = axs.flatten()
    for i, name in enumerate(fields):
        ax = axs_flat[i]
        v = samples[name]
        tri = ax.tricontourf(x, y, v, levels=18, cmap="viridis")
        fig.colorbar(tri, ax=ax, shrink=0.85)
        spread = float(np.max(v) - np.min(v))
        err = float(np.max(np.abs(v - expected[name])))
        ax.set_title(f"{name}\nspread={spread:.3e}, max|err|={err:.3e}")
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


def plot_eigenvalues(lam: np.ndarray, near_zero_count: int, outdir: Path) -> Path:
    path = outdir / "distorted_single_element_eigenvalues.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(1, lam.size + 1)
    lam_nonneg = np.clip(lam, 1.0e-18, None)
    ax.semilogy(idx, lam_nonneg, "o-", lw=1.3, ms=4)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title(f"Unsupported distorted element spectrum (near-zero count: {near_zero_count})")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
