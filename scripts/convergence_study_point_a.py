"""
Gmsh-only convergence study: tip deflection magnitude at point A vs mesh refinement.

Saves:
  - convergence curve: ``convergence_point_a_gmsh.png``
  - one mesh PNG per level: ``meshes_by_level/gmsh/Lxx_res_<r>/mesh_nel_<N>.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plate_fea.mesh import HeterosisMesh
from plate_fea.plotting import plot_heterosis_mesh
from plate_fea.problem_orchestrator import ProblemConfig, ProblemResult, solve_plate_problem


OUT_CONVERGENCE = "convergence_point_a_gmsh"
MESHES_BY_LEVEL_ROOT = "meshes_by_level/gmsh"
DEFAULT_OUT_DIR = Path("output/convergence")
DEFAULT_RESOLUTIONS = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8)
LENGTH_UNIT_LABEL = "mm"

REFINEMENT_COLOR_COARSE: tuple[float, float, float] = (0.20, 0.45, 0.62)
REFINEMENT_COLOR_FINE: tuple[float, float, float] = (0.78, 0.42, 0.22)


def _refinement_colors(n: int) -> list[tuple[float, float, float]]:
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


def _problem_config_resolution(base: ProblemConfig, resolution: int) -> ProblemConfig:
    return ProblemConfig(
        geometry=base.geometry,
        mesh_strategy="gmsh_boundary_sensitive",
        resolution=int(resolution),
        hole_refine=base.hole_refine,
        buffer=base.buffer,
        young_modulus=base.young_modulus,
        poisson_ratio=base.poisson_ratio,
        thickness=base.thickness,
        hole_top_shear_load=base.hole_top_shear_load,
        clamped_outer_edges=base.clamped_outer_edges,
        tolerance=base.tolerance,
    )


def _level_subdir(level_1based: int, resolution: int) -> str:
    r = int(resolution)
    slug = f"m{abs(r)}" if r < 0 else str(r)
    return f"L{level_1based:02d}_res_{slug}"


def _save_mesh_level_png(
    *,
    mesh: HeterosisMesh,
    out_path: Path,
    title: str,
    edge_color: tuple[float, float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_matlab_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.85))
    plot_heterosis_mesh(
        mesh,
        ax=ax,
        show_w_nodes=False,
        show_theta_nodes=False,
        show_q9_center_nodes=False,
        element_edge_color=edge_color,
        title=None,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$x$ $\mathrm{(mm)}$")
    ax.set_ylabel(r"$y$ $\mathrm{(mm)}$")
    ax.set_title(title, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.12)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def run_study(
    *,
    base_config: ProblemConfig,
    resolutions: tuple[int, ...],
    out_dir: Path,
    length_unit: str,
    show: bool,
) -> tuple[ProblemResult, int]:
    if len(resolutions) < 2:
        raise ValueError("Need at least two resolution values for a convergence study.")

    print(f"\n=== convergence study ({len(resolutions)} mesh levels), gmsh boundary sensitive ===")

    meshes: list[HeterosisMesh] = []
    w_a: list[float] = []
    n_el: list[int] = []
    n_dof: list[int] = []
    labels: list[str] = []
    wall_s: list[float] = []
    last_result: ProblemResult | None = None

    for r in resolutions:
        cfg = _problem_config_resolution(base_config, int(r))
        t0 = perf_counter()
        result = solve_plate_problem(cfg)
        wall_s.append(float(perf_counter() - t0))
        last_result = result
        m = result.model.mesh
        meshes.append(m)
        w_a.append(result.point_a_deflection)
        n_el.append(m.total_element_number)
        n_dof.append(m.total_dof_number)
        labels.append(rf"$N_{{\mathrm{{el}}}}={m.total_element_number}$ (${wall_s[-1]:.2f}\,\mathrm{{s}}$)")

    assert last_result is not None

    w_a_arr = np.asarray(w_a, dtype=float)
    n_el_arr = np.asarray(n_el, dtype=float)
    sort_idx = np.argsort(n_el_arr)
    rank_coarse_to_fine = np.argsort(np.argsort(n_el_arr))
    palette = _refinement_colors(len(resolutions))
    colors = [palette[int(r)] for r in rank_coarse_to_fine]
    out_dir.mkdir(parents=True, exist_ok=True)

    w_a_micrometre = np.abs(w_a_arr) * 1000.0

    apply_matlab_plot_style()
    fig_c, ax_c = plt.subplots(figsize=(7.2, 5.0))
    for ne_i, w_um, lab, c in zip(n_el_arr, w_a_micrometre, labels, colors, strict=True):
        ax_c.scatter([ne_i], [w_um], s=70, c=[c], edgecolors="black", linewidths=0.6, zorder=3, label=lab)

    ax_c.plot(
        n_el_arr[sort_idx],
        w_a_micrometre[sort_idx],
        color=(0.45, 0.45, 0.45),
        linestyle="-",
        linewidth=1.0,
        zorder=1,
    )
    ax_c.set_xlabel(r"Number of heterosis plate elements $N_{\mathrm{el}}$")
    ax_c.set_ylabel(r"Tip deflection magnitude $|w_A|$ ($\mu\mathrm{m}$)")
    ax_c.set_title(r"Convergence: $|w_A|$ at corner A vs mesh refinement (Gmsh boundary-sensitive)")
    ax_c.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_c.tick_params(axis="y", which="major", pad=8)
    ax_c.legend(loc="best", framealpha=0.95)
    fig_c.subplots_adjust(left=0.20, right=0.97, top=0.92, bottom=0.14)
    conv_path = out_dir / f"{OUT_CONVERGENCE}.png"
    fig_c.savefig(conv_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig_c)

    all_xy = np.vstack([m.node_coordinates for m in meshes])
    span = float(np.max(all_xy.max(axis=0) - all_xy.min(axis=0)))
    pad = 0.02 * span if span > 0 else 1.0
    xlim = (float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad))
    ylim = (float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad))

    mesh_root = out_dir / MESHES_BY_LEVEL_ROOT
    mesh_saved: list[Path] = []
    for lev, (mesh, c, r_i) in enumerate(zip(meshes, colors, resolutions, strict=True), start=1):
        sub = mesh_root / _level_subdir(lev, r_i)
        out_p = sub / f"mesh_nel_{n_el[lev - 1]}.png"
        _save_mesh_level_png(
            mesh=mesh,
            out_path=out_p,
            title=f"N_el={n_el[lev - 1]}  (t={wall_s[lev - 1]:.2f} s)",
            edge_color=c,
            xlim=xlim,
            ylim=ylim,
            show=show,
        )
        mesh_saved.append(out_p)

    for r_i, w, ne, nd, dt in zip(resolutions, w_a, n_el, n_dof, wall_s, strict=True):
        print(f"  res={r_i}:  w_A = {w:.8e} {length_unit}  (N_el = {ne}, DOF = {nd}, wall time = {dt:.3f} s)")
    w_sorted = w_a_arr[sort_idx]
    rel_change = abs(w_sorted[-1] - w_sorted[-2]) / max(abs(w_sorted[-1]), 1e-30)
    print(f"  Relative change |w_last - w_prev| / |w_last|: {rel_change:.4e}")
    print(f"  Total wall time (all solves): {sum(wall_s):.3f} s")
    print(f"  Saved: {conv_path}")
    print(f"  Saved {len(mesh_saved)} mesh figure(s) under: {mesh_root}")

    return last_result, max(resolutions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gmsh convergence study: tip deflection at point A.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for PNG figures (created if missing); default {DEFAULT_OUT_DIR!s}.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")
    parser.add_argument(
        "--resolutions",
        type=str,
        default=",".join(str(r) for r in DEFAULT_RESOLUTIONS),
        help="Comma-separated mesh refinement indices (default 10 levels: -1..8). Need at least two values.",
    )
    args = parser.parse_args()

    res_tuple = tuple(int(x.strip()) for x in args.resolutions.split(",") if x.strip())
    if len(res_tuple) < 2:
        parser.error("--resolutions must contain at least two integers.")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = ProblemConfig()

    run_study(
        base_config=base_config,
        resolutions=res_tuple,
        out_dir=out_dir,
        length_unit=LENGTH_UNIT_LABEL,
        show=args.show,
    )


if __name__ == "__main__":
    main()
