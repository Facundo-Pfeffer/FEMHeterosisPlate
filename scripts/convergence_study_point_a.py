"""
Convergence study: tip deflection at point A (hole corner) vs mesh refinement.

**Unit convention (default, consistent with ``ProblemConfig`` geometry in mm):**
plan dimensions and ``w`` in **mm**, ``E`` in **MPa** (N/mm²), plate thickness in **mm**,
hole-top line load ``q`` in **N/mm** (assignment default **1 kN/mm** → ``1000`` N/mm magnitude, sign for direction).

**Convergence:** one plot for all eight ``resolution`` values (default ``-1,0,1,2,3,4,5,6`` —
one step coarser than ``0..7``; ``-1`` is the coarsest supported buffer-ring level).
Abscissa: ``N_{\\mathrm{el}}`` (one heterosis plate element per mesh cell); ordinate: ``w_A`` in **µm**
(three decimals). Each solve is wall-timed. One mesh gallery figure shows all eight levels.

Example (from repo root, after ``pip install -e .``):

    python scripts/convergence_study_point_a.py
    python scripts/convergence_study_point_a.py --out-dir output --show

Uses ``ProblemConfig`` defaults. ``--resolutions`` must list **exactly eight** integers (all appear on
the same convergence curve; mesh gallery uses the same order)."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from plate_fea.mesh import HeterosisMesh
from plate_fea.plotting import plot_heterosis_mesh
from plate_fea.problem_orchestrator import ProblemConfig, ProblemResult, solve_plate_problem

# Output basenames (``.png`` added when saving).
OUT_CONVERGENCE = "convergence_point_a"
OUT_MESHES = "meshes_eight_levels"
DEFAULT_RESOLUTIONS_8 = (-1, 0, 1, 2, 3, 4, 5, 6)
LENGTH_UNIT_LABEL = "mm"

# Sequential marker / mesh-edge colours: interpolate in sRGB from coarse (cool) to fine (warm).
# Endpoints are muted for print; progression reads as refinement, not unrelated categories.
REFINEMENT_COLOR_COARSE: tuple[float, float, float] = (0.20, 0.45, 0.62)  # steel blue
REFINEMENT_COLOR_FINE: tuple[float, float, float] = (0.78, 0.42, 0.22)  # burnt sienna


def _refinement_colors(n: int) -> list[tuple[float, float, float]]:
    """``n`` colours along a straight blend coarse → fine (inclusive endpoints)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [REFINEMENT_COLOR_COARSE]
    c0 = np.asarray(REFINEMENT_COLOR_COARSE, dtype=float)
    c1 = np.asarray(REFINEMENT_COLOR_FINE, dtype=float)
    t = np.linspace(0.0, 1.0, n, dtype=float)[:, np.newaxis]
    rgb = np.clip((1.0 - t) * c0 + t * c1, 0.0, 1.0)
    return [tuple(float(x) for x in row) for row in rgb]


def apply_matlab_plot_style() -> None:
    """Clean figure defaults (grid, sans-serif) suitable for reports."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )


def _figure_with_bottom_caption(
    fig: Figure,
    *,
    caption: str,
    bottom_margin: float = 0.26,
    left: float = 0.14,
    right: float = 0.97,
    top: float = 0.93,
) -> None:
    """Reserve space under axes so the caption never overlaps axis labels."""
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom_margin)
    fig.text(
        0.5,
        0.02,
        caption,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.32",
        linespacing=1.35,
    )


def run_study(
    *,
    base_config: ProblemConfig,
    resolutions: tuple[int, ...],
    out_dir: Path,
    convergence_name: str,
    meshes_name: str,
    length_unit: str,
    show: bool,
) -> tuple[ProblemResult, int]:
    if len(resolutions) < 2:
        raise ValueError("Need at least two resolution values for a convergence study.")

    print(f"\n=== convergence study ({len(resolutions)} mesh levels) ===")

    meshes: list[HeterosisMesh] = []
    w_a: list[float] = []
    n_el: list[int] = []
    n_dof: list[int] = []
    labels: list[str] = []
    wall_s: list[float] = []
    last_result: ProblemResult | None = None

    for r in resolutions:
        cfg = ProblemConfig(
            geometry=base_config.geometry,
            resolution=int(r),
            hole_refine=base_config.hole_refine,
            buffer=base_config.buffer,
            young_modulus=base_config.young_modulus,
            poisson_ratio=base_config.poisson_ratio,
            thickness=base_config.thickness,
            hole_top_shear_load=base_config.hole_top_shear_load,
            clamped_outer_edges=base_config.clamped_outer_edges,
            tolerance=base_config.tolerance,
        )
        t0 = perf_counter()
        result = solve_plate_problem(cfg)
        wall_s.append(float(perf_counter() - t0))
        last_result = result
        m = result.model.mesh
        meshes.append(m)
        w_a.append(result.point_a_deflection)
        n_el.append(m.total_element_number)
        n_dof.append(m.total_dof_number)
        dt = wall_s[-1]
        labels.append(rf"$\mathrm{{res}}={int(r)}$ (${dt:.2f}\,\mathrm{{s}}$)")

    assert last_result is not None

    w_a_arr = np.asarray(w_a, dtype=float)
    n_el_arr = np.asarray(n_el, dtype=float)
    sort_idx = np.argsort(n_el_arr)
    # Map each case to a colour by mesh size (coarsest → finest), not by CLI order.
    rank_coarse_to_fine = np.argsort(np.argsort(n_el_arr))
    palette = _refinement_colors(len(resolutions))
    colors = [palette[int(r)] for r in rank_coarse_to_fine]

    out_dir.mkdir(parents=True, exist_ok=True)

    # $w_A$ is $\mathcal{O}(10^{-4})$ mm; three decimals in mm would collapse every tick to $-0.000$.
    w_a_micrometre = w_a_arr * 1000.0

    unit_note = (
        r"Units: length $\mathrm{mm}$, $E$ $\mathrm{MPa}$, line load $q$ $\mathrm{N/mm}$, "
        r"deflection $w$ $\mathrm{mm}$, resultant $R$ $\mathrm{N}$. "
        r"Convergence $y$-axis: $w_A$ in $\mu\mathrm{m}$ ($w/\mathrm{mm}\times 10^3$)."
    )

    # --- Figure 1: convergence ---
    apply_matlab_plot_style()
    fig_c, ax_c = plt.subplots(figsize=(7.2, 5.0))

    for ne_i, w_um, lab, c in zip(n_el_arr, w_a_micrometre, labels, colors, strict=True):
        ax_c.scatter(
            [ne_i],
            [w_um],
            s=70,
            c=[c],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label=lab,
        )

    ax_c.plot(
        n_el_arr[sort_idx],
        w_a_micrometre[sort_idx],
        color=(0.45, 0.45, 0.45),
        linestyle="-",
        linewidth=1.0,
        zorder=1,
    )
    ax_c.set_xlabel(r"Number of heterosis plate elements $N_{\mathrm{el}}$")
    ax_c.set_ylabel(r"Tip deflection $w_A$ ($\mu\mathrm{m}$)")
    ax_c.set_title(r"Convergence: $w_A$ at corner A vs mesh refinement")
    ax_c.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_c.tick_params(axis="y", which="major", pad=8)
    ax_c.legend(loc="best", framealpha=0.95)
    _figure_with_bottom_caption(fig_c, caption=unit_note, bottom_margin=0.26, top=0.92, left=0.20)
    conv_path = out_dir / f"{convergence_name}.png"
    fig_c.savefig(conv_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig_c)

    # --- Figure 2: mesh gallery ---
    apply_matlab_plot_style()
    n_m = len(meshes)
    ncols = 4 if n_m > 4 else (2 if n_m > 1 else 1)
    nrows = (n_m + ncols - 1) // ncols
    fig_m, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 3.75 * nrows),
    )
    ax_flat = np.atleast_1d(axes).ravel()

    all_xy = np.vstack([m.node_coordinates for m in meshes])
    span = float(np.max(all_xy.max(axis=0) - all_xy.min(axis=0)))
    pad = 0.02 * span if span > 0 else 1.0
    xlim = (float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad))
    ylim = (float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad))

    for idx, (mesh, lab, c) in enumerate(zip(meshes, labels, colors, strict=True)):
        ax = ax_flat[idx]
        plot_heterosis_mesh(
            mesh,
            ax=ax,
            show_w_nodes=False,
            show_theta_nodes=False,
            element_edge_color=c,
            title=lab,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$x$ $\mathrm{(mm)}$")
        ax.set_ylabel(r"$y$ $\mathrm{(mm)}$")

    for j in range(len(meshes), len(ax_flat)):
        ax_flat[j].set_visible(False)

    fig_m.suptitle(
        rf"Heterosis plate meshes ($N={n_m}$ levels; edge color: coarse $\rightarrow$ fine)",
        fontsize=13,
        y=0.995,
    )
    fig_m.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.20, hspace=0.35, wspace=0.28)
    fig_m.text(
        0.5,
        0.02,
        unit_note,
        transform=fig_m.transFigure,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.32",
        linespacing=1.35,
    )
    mesh_path = out_dir / f"{meshes_name}.png"
    fig_m.savefig(mesh_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig_m)

    for r_i, w, ne, nd, dt in zip(resolutions, w_a, n_el, n_dof, wall_s, strict=True):
        print(
            f"  res={r_i}:  w_A = {w:.8e} {length_unit}  "
            f"(N_el = {ne}, DOF = {nd}, wall time = {dt:.3f} s)"
        )
    w_sorted = w_a_arr[sort_idx]
    rel_change = abs(w_sorted[-1] - w_sorted[-2]) / max(abs(w_sorted[-1]), 1e-30)
    print(f"  Relative change |w_last - w_prev| / |w_last|: {rel_change:.4e}")
    print(f"  Total wall time (all solves): {sum(wall_s):.3f} s")
    print(f"  Saved: {conv_path}")
    print(f"  Saved: {mesh_path}")

    return last_result, max(resolutions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convergence study: tip deflection at point A (ProblemConfig defaults).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for PNG figures (created if missing).",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    parser.add_argument(
        "--resolutions",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RESOLUTIONS_8),
        help="Exactly eight integers (mesh refinement indices); all on one convergence plot.",
    )
    args = parser.parse_args()

    res_tuple = tuple(int(x.strip()) for x in args.resolutions.split(",") if x.strip())
    if len(res_tuple) != 8:
        parser.error("--resolutions must list exactly eight integers.")

    base_config = ProblemConfig()

    run_study(
        base_config=base_config,
        resolutions=res_tuple,
        out_dir=args.out_dir,
        convergence_name=OUT_CONVERGENCE,
        meshes_name=OUT_MESHES,
        length_unit=LENGTH_UNIT_LABEL,
        show=args.show,
    )


if __name__ == "__main__":
    main()
