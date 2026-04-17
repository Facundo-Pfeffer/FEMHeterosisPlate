"""
Convergence comparison with matched mesh sizes:
baseline uniform buffer ring vs gmsh boundary-sensitive.

Goal:
  - eight baseline uniform-buffer levels: **L1** from a **joint** search (default) — coarsest pair
    ``(uniform, gmsh)`` whose ``N_el`` differ by at most ``--l1-max-rel-nel`` (relative to ``max(N_el)``);
 **L2..L8** use ``--baseline-resolutions`` entries 2..8 with ``ProblemConfig.hole_refine``.
  - Legacy modes: ``--l1-pairing fixed`` (L1 uniform = first resolution + ``--baseline-coarsest-hole-refine``,
    gmsh same ``(res,hr)``) or ``gmsh_nel`` (gmsh L1 = closest ``N_el`` to uniform L1).
  - gmsh levels 2..8: match increasing baseline ``N_el`` with strictly increasing gmsh ``N_el``,
  - compare tip deflection at A vs N_el; ordinate in mm.
  - saves two 4×2 mesh gallery PNGs (uniform and gmsh, portrait) unless ``--no-mesh-gallery``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

from plate_fea.mesh import HeterosisMesh
from plate_fea.plotting import plot_heterosis_mesh
from plate_fea.problem_orchestrator import ProblemConfig, generate_mesh, solve_plate_problem


DEFAULT_BASELINE_RESOLUTIONS_8 = (1, 2, 3, 4, 5, 6, 7, 8)
OUT_CONVERGENCE = "convergence_uniform_vs_gmsh_matched_8levels"
OUT_MATCHING = "mesh_count_matching_uniform_vs_gmsh_8levels"
OUT_MESH_GALLERY = "mesh_gallery_uniform_vs_gmsh_8levels"

# Mesh gallery: edge colour coarse → fine (same convention as ``convergence_study_point_a``).
_REF_COARSE: tuple[float, float, float] = (0.20, 0.45, 0.62)
_REF_FINE: tuple[float, float, float] = (0.78, 0.42, 0.22)


def _refinement_colors(n: int) -> list[tuple[float, float, float]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [_REF_COARSE]
    c0 = np.asarray(_REF_COARSE, dtype=float)
    c1 = np.asarray(_REF_FINE, dtype=float)
    t = np.linspace(0.0, 1.0, n, dtype=float)[:, np.newaxis]
    rgb = np.clip((1.0 - t) * c0 + t * c1, 0.0, 1.0)
    return [tuple(float(x) for x in row) for row in rgb]


@dataclass(frozen=True)
class CaseRun:
    label: str
    config: ProblemConfig
    mesh: HeterosisMesh
    n_el: int
    n_dof: int
    w_a_mm: float
    wall_s: float


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )


def _mesh_count(config: ProblemConfig) -> tuple[int, int]:
    mesh = generate_mesh(config)
    return mesh.total_element_number, mesh.total_dof_number


def _solve_case(label: str, config: ProblemConfig) -> CaseRun:
    t0 = perf_counter()
    result = solve_plate_problem(config)
    dt = float(perf_counter() - t0)
    return CaseRun(
        label=label,
        config=config,
        mesh=result.model.mesh,
        n_el=result.model.mesh.total_element_number,
        n_dof=result.model.mesh.total_dof_number,
        w_a_mm=float(result.point_a_deflection),
        wall_s=dt,
    )


def _uniform_problem_config(base: ProblemConfig, resolution: int, hole_refine: int) -> ProblemConfig:
    return ProblemConfig(
        geometry=base.geometry,
        mesh_strategy="uniform_buffer_ring",
        resolution=int(resolution),
        hole_refine=int(hole_refine),
        buffer=base.buffer,
        young_modulus=base.young_modulus,
        poisson_ratio=base.poisson_ratio,
        thickness=base.thickness,
        clamped_outer_edges=base.clamped_outer_edges,
        hole_top_shear_load=base.hole_top_shear_load,
        tolerance=base.tolerance,
    )


def _build_baseline_configs(
    base: ProblemConfig,
    resolutions: tuple[int, ...],
    *,
    coarsest_hole_refine: int,
) -> list[ProblemConfig]:
    """First level uses ``coarsest_hole_refine``; rest use ``base.hole_refine``."""
    if coarsest_hole_refine < 0:
        raise ValueError("coarsest_hole_refine must be >= 0")
    out: list[ProblemConfig] = []
    for i, r in enumerate(resolutions):
        hr = int(coarsest_hole_refine) if i == 0 else int(base.hole_refine)
        out.append(_uniform_problem_config(base, int(r), hr))
    return out


def _build_uniform_candidate_pool(base: ProblemConfig, res_max: int, hr_max: int) -> list[tuple[int, int, int, int]]:
    """Candidate tuples ``(resolution, hole_refine, n_el, n_dof)`` for ``uniform_buffer_ring``."""
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(-1, res_max + 1):
        for hr in range(0, hr_max + 1):
            cfg = _uniform_problem_config(base, r, hr)
            n_el, n_dof = _mesh_count(cfg)
            candidates.append((r, hr, n_el, n_dof))
    candidates.sort(key=lambda t: (t[2], t[0], t[1]))
    return candidates


def _rel_nel_diff(nu: int, ng: int) -> float:
    mx = max(int(nu), int(ng), 1)
    return abs(float(nu) - float(ng)) / float(mx)


def _pick_coarsest_matched_l1_pair(
    uniform_pool: list[tuple[int, int, int, int]],
    gmsh_pool: list[tuple[int, int, int, int]],
    *,
    max_rel_nel: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
    """
    Coarsest pair ``(uniform, gmsh)`` with comparable ``N_el``.

    Prefer pairs with ``|N_u - N_g| / max(N_u, N_g) <= max_rel_nel``; among those, minimize
    ``max(N_u, N_g)`` (coarsest common scale), then relative gap.

    If none meet the tolerance, fall back to minimizing the relative gap, then ``max(N_u, N_g)``.

    Returns ``(u_tuple, g_tuple, used_fallback)``.
    """
    best_in_tol: tuple[int, float, tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
    # (max_nel, rel, u, g)
    for u in uniform_pool:
        nu = u[2]
        for g in gmsh_pool:
            ng = g[2]
            rel = _rel_nel_diff(nu, ng)
            if rel > max_rel_nel + 1e-12:
                continue
            mx = max(nu, ng)
            if best_in_tol is None:
                best_in_tol = (mx, rel, u, g)
            else:
                mx0, rel0, _, _ = best_in_tol
                if mx < mx0 - 1e-9 or (abs(mx - mx0) <= 1e-9 and rel < rel0 - 1e-9):
                    best_in_tol = (mx, rel, u, g)
    if best_in_tol is not None:
        return best_in_tol[2], best_in_tol[3], False

    best_u: tuple[int, int, int, int] | None = None
    best_g: tuple[int, int, int, int] | None = None
    best_rel = float("inf")
    best_mx = float("inf")
    for u in uniform_pool:
        nu = u[2]
        for g in gmsh_pool:
            ng = g[2]
            rel = _rel_nel_diff(nu, ng)
            mx = max(nu, ng)
            if rel < best_rel - 1e-9 or (abs(rel - best_rel) <= 1e-9 and mx < best_mx - 1e-9):
                best_rel = rel
                best_mx = mx
                best_u, best_g = u, g
    if best_u is None or best_g is None:
        raise RuntimeError("Empty uniform or gmsh candidate pool for L1 pairing.")
    return best_u, best_g, True


def _build_gmsh_candidate_pool(base: ProblemConfig, res_max: int, hr_max: int) -> list[tuple[int, int, int, int]]:
    """
    Return candidate tuples: (resolution, hole_refine, n_el, n_dof).
    """
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(-1, res_max + 1):
        for hr in range(0, hr_max + 1):
            cfg = ProblemConfig(
                geometry=base.geometry,
                mesh_strategy="gmsh_boundary_sensitive",
                resolution=int(r),
                hole_refine=int(hr),
                buffer=base.buffer,
                young_modulus=base.young_modulus,
                poisson_ratio=base.poisson_ratio,
                thickness=base.thickness,
                clamped_outer_edges=base.clamped_outer_edges,
                hole_top_shear_load=base.hole_top_shear_load,
                tolerance=base.tolerance,
            )
            n_el, n_dof = _mesh_count(cfg)
            candidates.append((r, hr, n_el, n_dof))
    # unique by (r, hr) already, sort by element count for matching
    candidates.sort(key=lambda t: (t[2], t[0], t[1]))
    return candidates


def _match_gmsh_to_targets(
    targets_nel: list[int],
    candidates: list[tuple[int, int, int, int]],
    *,
    fixed_first_rr_hr: tuple[int, int] | None,
) -> list[tuple[int, int, int, int]]:
    """
    Level 1: either exact ``(resolution, hole_refine)`` from ``fixed_first_rr_hr`` (same as baseline L1),
    or (if ``fixed_first_rr_hr is None``) gmsh candidate whose N_el is closest to baseline L1 target.
    Later levels: closest match to each target among unused candidates with strictly increasing N_el.
    """
    if not targets_nel:
        return []
    chosen: list[tuple[int, int, int, int]] = []
    used: set[tuple[int, int]] = set()

    best_first: tuple[int, int, int, int] | None = None
    if fixed_first_rr_hr is not None:
        r_fix, hr_fix = int(fixed_first_rr_hr[0]), int(fixed_first_rr_hr[1])
        for r, hr, n_el, n_dof in candidates:
            if (r, hr) == (r_fix, hr_fix):
                best_first = (r, hr, n_el, n_dof)
                break
        if best_first is None:
            raise RuntimeError(
                f"No gmsh candidate with resolution={r_fix}, hole_refine={hr_fix} in pool; "
                "increase --gmsh-res-max / --gmsh-hole-refine-max or adjust baseline L1.",
            )
    else:
        t0 = targets_nel[0]
        best_abs = float("inf")
        for r, hr, n_el, n_dof in candidates:
            err = abs(float(n_el) - float(t0))
            if err < best_abs - 1e-9:
                best_abs = err
                best_first = (r, hr, n_el, n_dof)
            elif abs(err - best_abs) <= 1e-9 and best_first is not None:
                if (r, hr) < (best_first[0], best_first[1]):
                    best_first = (r, hr, n_el, n_dof)
        if best_first is None:
            raise RuntimeError("Empty gmsh candidate pool.")
    chosen.append(best_first)
    used.add((best_first[0], best_first[1]))
    last_nel = best_first[2]

    for target in targets_nel[1:]:
        best_idx = -1
        best_score = float("inf")
        for i, (r, hr, n_el, _n_dof) in enumerate(candidates):
            if (r, hr) in used:
                continue
            if n_el <= last_nel:
                continue
            rel = abs(n_el - target) / max(target, 1)
            overshoot_penalty = 0.15 if n_el > target else 0.0
            score = rel + overshoot_penalty
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            # Still enforce strictly increasing N_el (do not relax monotonicity).
            for i, (r, hr, n_el, _n_dof) in enumerate(candidates):
                if (r, hr) in used:
                    continue
                if n_el <= last_nel:
                    continue
                rel = abs(n_el - target) / max(target, 1)
                if rel < best_score:
                    best_score = rel
                    best_idx = i
        if best_idx < 0:
            # Last resort: coarsest unused mesh that is still finer than the previous level.
            bump_idx = -1
            bump_nel = None
            for i, (r, hr, n_el, _n_dof) in enumerate(candidates):
                if (r, hr) in used:
                    continue
                if n_el <= last_nel:
                    continue
                if bump_nel is None or n_el < bump_nel:
                    bump_nel = n_el
                    bump_idx = i
            if bump_idx < 0:
                raise RuntimeError(
                    "No gmsh candidate left with N_el greater than the previous level; "
                    "try increasing --gmsh-res-max or --gmsh-hole-refine-max.",
                )
            best_idx = bump_idx

        pick = candidates[best_idx]
        chosen.append(pick)
        used.add((pick[0], pick[1]))
        last_nel = pick[2]

    return chosen


def _save_mesh_gallery_one_strategy(
    *,
    runs: list[CaseRun],
    study_resolutions: tuple[int, ...],
    palette: list[tuple[float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    out_path: Path,
    banner_text: str,
    suptitle: str,
    show: bool,
) -> None:
    """Single strategy: 8 meshes in a 4 (rows) × 2 (columns) portrait figure; L1,L2 | row1 … L7,L8 | row4."""
    mesh_rows = 4
    banner_h = 0.11
    height_ratios = [banner_h] + [1.0] * mesh_rows
    fig = plt.figure(figsize=(8.2, 14.5))
    gs = GridSpec(
        len(height_ratios),
        2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.34,
        wspace=0.22,
        left=0.10,
        right=0.97,
        top=0.96,
        bottom=0.04,
    )
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.axis("off")
    ax_banner.text(0.5, 0.4, banner_text, ha="center", va="center", fontsize=11.5, fontweight="bold")

    for j in range(8):
        br = runs[j]
        bc = br.config
        c = palette[j]
        sres = int(study_resolutions[j])
        ax = fig.add_subplot(gs[1 + j // 2, j % 2])
        plot_heterosis_mesh(
            br.mesh,
            ax=ax,
            show_w_nodes=False,
            show_theta_nodes=False,
            show_q9_center_nodes=False,
            element_edge_color=c,
            title=None,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$x$ (mm)", fontsize=9)
        ax.set_ylabel(r"$y$ (mm)", fontsize=9)
        ttl = (
            rf"L{j+1}  study $\mathrm{{res}}={sres}$"
            + "\n"
            + rf"mesh $(\mathrm{{res}},\mathrm{{hr}})=({bc.resolution},{bc.hole_refine})$  "
            + rf"$N_{{\mathrm{{el}}}}={br.n_el}$"
            + "\n"
            + rf"$t={br.wall_s:.2f}\,\mathrm{{s}}$"
        )
        ax.set_title(ttl, fontsize=8.5)

    fig.suptitle(suptitle, fontsize=12.0, y=0.995)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def _save_mesh_gallery(
    *,
    baseline_runs: list[CaseRun],
    gmsh_runs: list[CaseRun],
    study_resolutions: tuple[int, ...],
    out_dir: Path,
    gallery_basename: str,
    show: bool,
) -> tuple[Path, Path]:
    """Two PNG files (uniform and gmsh), each 4×2 portrait; shared axis limits from all meshes."""
    n = len(baseline_runs)
    if n != 8 or len(gmsh_runs) != 8:
        raise ValueError("Mesh gallery expects 8 baseline and 8 gmsh runs.")
    if len(study_resolutions) != 8:
        raise ValueError("study_resolutions must have length 8.")
    palette = _refinement_colors(n)
    all_meshes = [r.mesh for r in baseline_runs] + [r.mesh for r in gmsh_runs]
    all_xy = np.vstack([m.node_coordinates for m in all_meshes])
    span = float(np.max(all_xy.max(axis=0) - all_xy.min(axis=0)))
    pad = 0.02 * span if span > 0 else 1.0
    xlim = (float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad))
    ylim = (float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad))

    p_uniform = out_dir / f"{gallery_basename}_uniform.png"
    p_gmsh = out_dir / f"{gallery_basename}_gmsh.png"

    _save_mesh_gallery_one_strategy(
        runs=baseline_runs,
        study_resolutions=study_resolutions,
        palette=palette,
        xlim=xlim,
        ylim=ylim,
        out_path=p_uniform,
        banner_text=(
            r"uniform_buffer_ring — 8 meshes (4$\times$2: rows $(\mathrm{L1},\mathrm{L2})\ldots(\mathrm{L7},\mathrm{L8})$; "
            r"study res $1{\rightarrow}8$; $t$ = wall time for full solve)"
        ),
        suptitle=r"uniform_buffer_ring (edge colour coarse $\rightarrow$ fine)",
        show=show,
    )
    _save_mesh_gallery_one_strategy(
        runs=gmsh_runs,
        study_resolutions=study_resolutions,
        palette=palette,
        xlim=xlim,
        ylim=ylim,
        out_path=p_gmsh,
        banner_text=(
            r"gmsh_boundary_sensitive — 8 meshes (same level order; 4$\times$2 layout; "
            r"$t$ = wall time for full solve)"
        ),
        suptitle=r"gmsh_boundary_sensitive (edge colour coarse $\rightarrow$ fine)",
        show=show,
    )
    return (p_uniform, p_gmsh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="8-level convergence comparison with element-count matching (uniform vs gmsh).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory for PNG figures.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument(
        "--baseline-resolutions",
        type=str,
        default=",".join(str(r) for r in DEFAULT_BASELINE_RESOLUTIONS_8),
        help="Exactly 8 baseline uniform-buffer resolutions (default: 1..8).",
    )
    parser.add_argument(
        "--gmsh-res-max",
        type=int,
        default=12,
        help="Max gmsh resolution scanned in candidate search (must cover finest baseline N_el).",
    )
    parser.add_argument(
        "--gmsh-hole-refine-max",
        type=int,
        default=12,
        help="Max gmsh hole_refine scanned in candidate search (must cover finest baseline N_el).",
    )
    parser.add_argument(
        "--no-mesh-gallery",
        action="store_true",
        help="Do not save mesh gallery PNGs (two 4×2 portrait figures: *_uniform.png, *_gmsh.png).",
    )
    parser.add_argument(
        "--baseline-coarsest-hole-refine",
        type=int,
        default=0,
        help="With --l1-pairing fixed_rr or gmsh_nel: hole_refine for L1 uniform only. Ignored for joint.",
    )
    parser.add_argument(
        "--l1-pairing",
        choices=("joint", "fixed_rr", "gmsh_nel"),
        default="joint",
        help="joint: coarsest uniform×gmsh with similar N_el (see --l1-max-rel-nel). "
        "fixed_rr: L1 gmsh (res,hr) = L1 uniform from first baseline resolution + --baseline-coarsest-hole-refine. "
        "gmsh_nel: L1 uniform fixed as in fixed_rr; gmsh L1 = closest N_el in pool.",
    )
    parser.add_argument(
        "--l1-max-rel-nel",
        type=float,
        default=0.05,
        help="joint mode only: max |N_u-N_g|/max(N_u,N_g) for L1 pair (default: 0.05 = 5%%).",
    )
    args = parser.parse_args()

    baseline_res = tuple(int(x.strip()) for x in args.baseline_resolutions.split(",") if x.strip())
    if len(baseline_res) != 8:
        parser.error("--baseline-resolutions must contain exactly 8 integers.")

    base = ProblemConfig()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    res_max = int(args.gmsh_res_max)
    hr_max = int(args.gmsh_hole_refine_max)
    print("\n=== Scanning candidate pools (mesh counts only: uniform + gmsh) ===")
    uniform_pool = _build_uniform_candidate_pool(base, res_max=res_max, hr_max=hr_max)
    pool = _build_gmsh_candidate_pool(base, res_max=res_max, hr_max=hr_max)

    fix_l1: tuple[int, int] | None
    l1_joint_fallback = False
    if args.l1_pairing == "joint":
        if args.l1_max_rel_nel <= 0.0:
            parser.error("--l1-max-rel-nel must be positive for joint pairing.")
        l1_u, l1_g, l1_joint_fallback = _pick_coarsest_matched_l1_pair(
            uniform_pool,
            pool,
            max_rel_nel=float(args.l1_max_rel_nel),
        )
        nu0, ng0 = l1_u[2], l1_g[2]
        rel0 = _rel_nel_diff(nu0, ng0)
        print(
            f"\n=== L1 joint pairing (coarsest comparable N_el; tol={100.0 * float(args.l1_max_rel_nel):.2f}%) ===",
        )
        print(
            f"  uniform: res={l1_u[0]}, hr={l1_u[1]}, N_el={nu0}  |  "
            f"gmsh: res={l1_g[0]}, hr={l1_g[1]}, N_el={ng0}  |  "
            f"rel |ΔN|/max(N)={100.0 * rel0:.2f}%",
        )
        if l1_joint_fallback:
            print(
                " (no pair met the tolerance; using best relative match — "
                "widen --pool via --gmsh-res-max/--gmsh-hole-refine-max or relax --l1-max-rel-nel)",
            )
        print(
            f"  Note: joint L1 picks internal (res,hr) for matched N_el; plots use study res {baseline_res[0]}..{baseline_res[7]} (default 1..8). "
            f"L2..L8 uniform use mesh res {baseline_res[1]}..{baseline_res[7]}.",
        )
        baseline_cfgs = [_uniform_problem_config(base, l1_u[0], l1_u[1])] + [
            _uniform_problem_config(base, int(r), int(base.hole_refine)) for r in baseline_res[1:]
        ]
        fix_l1 = (int(l1_g[0]), int(l1_g[1]))
    else:
        baseline_cfgs = _build_baseline_configs(
            base,
            baseline_res,
            coarsest_hole_refine=int(args.baseline_coarsest_hole_refine),
        )
        if args.l1_pairing == "fixed_rr":
            fix_l1 = (int(baseline_cfgs[0].resolution), int(baseline_cfgs[0].hole_refine))
        else:
            fix_l1 = None

    # 1) Baseline solves (define N_el targets).
    baseline_runs: list[CaseRun] = []
    print("\n=== Baseline (uniform_buffer_ring) ===")
    if args.l1_pairing != "joint":
        print(
            f"  L1: res={baseline_cfgs[0].resolution}, hole_refine={baseline_cfgs[0].hole_refine}; "
            f"later levels: hole_refine={base.hole_refine}",
        )
    for cfg in baseline_cfgs:
        run = _solve_case(f"res={cfg.resolution},hr={cfg.hole_refine}", cfg)
        baseline_runs.append(run)
        print(f"  {run.label}: N_el={run.n_el}, DOF={run.n_dof}, w_A={run.w_a_mm:.8e} mm, t={run.wall_s:.2f}s")

    targets_nel = [r.n_el for r in baseline_runs]

    # 2) Gmsh match to targets (L1 fixed for joint / fixed_rr).
    pool_max_nel = max(p[2] for p in pool)
    finest_target = max(targets_nel)
    if pool_max_nel < finest_target:
        raise RuntimeError(
            f"Gmsh candidate pool max N_el={pool_max_nel} is below finest baseline target N_el={finest_target}. "
            "Increase --gmsh-res-max and/or --gmsh-hole-refine-max so the pool can reach the finest uniform mesh.",
        )
    if args.l1_pairing == "gmsh_nel":
        print(
            f"\n=== Gmsh level 1: candidate closest to baseline N_el={targets_nel[0]} (uniform L1) ===",
        )
    elif args.l1_pairing == "fixed_rr":
        assert fix_l1 is not None
        print(
            f"\n=== Gmsh level 1: fixed to baseline L1 (res={fix_l1[0]}, hole_refine={fix_l1[1]}) ===",
        )
    picks = _match_gmsh_to_targets(targets_nel, pool, fixed_first_rr_hr=fix_l1)
    r0, hr0, n0, _dof0 = picks[0]
    print(
        f"  gmsh L1: res={r0}, hr={hr0}, N_el={n0} "
        f"(baseline L1 N_el={targets_nel[0]}, |ΔN_el|={abs(n0 - targets_nel[0])}).",
    )

    gmsh_cfgs: list[ProblemConfig] = []
    print("\n=== Matched gmsh configurations (levels 2..8 follow baseline N_el targets) ===")
    for i, ((r, hr, n_el, n_dof), target_nel) in enumerate(zip(picks, targets_nel, strict=True), start=1):
        rel = abs(n_el - target_nel) / max(target_nel, 1)
        print(
            f"  L{i}: target N_el={target_nel} -> gmsh(res={r}, hr={hr}) "
            f"N_el={n_el}, DOF={n_dof}, rel.diff={100.0 * rel:.2f}%"
        )
        if rel > 0.25:
            print(
                "    (warning: >25% N_el mismatch — gmsh may not reach this coarse a count; "
                "interpret this level accordingly.)",
            )
        gmsh_cfgs.append(
            ProblemConfig(
                geometry=base.geometry,
                mesh_strategy="gmsh_boundary_sensitive",
                resolution=int(r),
                hole_refine=int(hr),
                buffer=base.buffer,
                young_modulus=base.young_modulus,
                poisson_ratio=base.poisson_ratio,
                thickness=base.thickness,
                clamped_outer_edges=base.clamped_outer_edges,
                hole_top_shear_load=base.hole_top_shear_load,
                tolerance=base.tolerance,
            )
        )

    # 3) Solve matched gmsh cases.
    gmsh_runs: list[CaseRun] = []
    print("\n=== gmsh_boundary_sensitive (matched levels) ===")
    for i, cfg in enumerate(gmsh_cfgs, start=1):
        run = _solve_case(f"res={cfg.resolution},hr={cfg.hole_refine}", cfg)
        gmsh_runs.append(run)
        print(f"  L{i} {run.label}: N_el={run.n_el}, DOF={run.n_dof}, w_A={run.w_a_mm:.8e} mm, t={run.wall_s:.2f}s")

    # 4) Plots.
    _style()
    b_nel = np.asarray([r.n_el for r in baseline_runs], dtype=float)
    g_nel = np.asarray([r.n_el for r in gmsh_runs], dtype=float)
    b_w_mm = np.asarray([r.w_a_mm for r in baseline_runs], dtype=float)
    g_w_mm = np.asarray([r.w_a_mm for r in gmsh_runs], dtype=float)
    b_wall = np.asarray([r.wall_s for r in baseline_runs], dtype=float)
    g_wall = np.asarray([r.wall_s for r in gmsh_runs], dtype=float)
    idx_b = np.argsort(b_nel)
    idx_g = np.argsort(g_nel)

    fig1, ax1 = plt.subplots(figsize=(8.0, 5.3))
    (ln_bw,) = ax1.plot(
        b_nel[idx_b],
        b_w_mm[idx_b],
        "o-",
        color="#355C7D",
        lw=1.8,
        ms=5.5,
        label=r"uniform $w_A$",
        zorder=3,
    )
    (ln_gw,) = ax1.plot(
        g_nel[idx_g],
        g_w_mm[idx_g],
        "s-",
        color="#d62728",
        lw=1.8,
        ms=5.0,
        label=r"gmsh $w_A$",
        zorder=3,
    )
    ax1.set_xlabel(r"Number of heterosis plate elements $N_{\mathrm{el}}$")
    ax1.set_ylabel(r"Tip deflection $w_A$ (mm)")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax1.set_title(r"8-level convergence comparison with matched $N_{\mathrm{el}}$ (baseline res 1$\rightarrow$8)")

    ax1t = ax1.twinx()
    (ln_bt,) = ax1t.plot(
        b_nel[idx_b],
        b_wall[idx_b],
        "o--",
        color="#5a7aa0",
        lw=1.3,
        ms=4.0,
        alpha=0.95,
        label="uniform wall time",
        zorder=2,
    )
    (ln_gt,) = ax1t.plot(
        g_nel[idx_g],
        g_wall[idx_g],
        "s--",
        color="#e65555",
        lw=1.3,
        ms=3.8,
        alpha=0.95,
        label="gmsh wall time",
        zorder=2,
    )
    ax1t.set_ylabel(r"Solve wall time (s)")
    ax1t.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1t.tick_params(axis="y", labelcolor="0.25")

    ax1.legend(handles=[ln_bw, ln_gw, ln_bt, ln_gt], loc="best", framealpha=0.95, fontsize=9.0)
    fig1.subplots_adjust(left=0.12, right=0.88, top=0.91, bottom=0.21)
    fig1.text(
        0.5,
        0.03,
        "L1: jointly coarsest comparable N_el (joint mode) or fixed pairing; later levels: matched to baseline N_el.",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="0.30",
    )
    p1 = out_dir / f"{OUT_CONVERGENCE}.png"
    fig1.savefig(p1, dpi=230)
    if args.show:
        plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8.2, 5.0))
    levels = np.arange(1, 9)
    ax2.plot(levels, b_nel, "o-", color="#355C7D", lw=1.7, ms=5.5, label="baseline N_el")
    ax2.plot(levels, g_nel, "s-", color="#d62728", lw=1.7, ms=5.0, label="gmsh matched N_el")
    for k in range(8):
        ax2.plot([levels[k], levels[k]], [b_nel[k], g_nel[k]], color="0.75", lw=1.0, zorder=1)
    ax2.set_xlabel(r"Study refinement level (res $1\rightarrow 8$, coarse $\rightarrow$ fine)")
    ax2.set_ylabel(r"Element count $N_{\mathrm{el}}$")
    ax2.set_title(r"Element-count matching quality per level")
    ax2.legend(loc="best", framealpha=0.95)
    fig2.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.14)
    p2 = out_dir / f"{OUT_MATCHING}.png"
    fig2.savefig(p2, dpi=230)
    if args.show:
        plt.show()
    plt.close(fig2)

    p3: tuple[Path, Path] | None = None
    if not args.no_mesh_gallery:
        p3 = _save_mesh_gallery(
            baseline_runs=baseline_runs,
            gmsh_runs=gmsh_runs,
            study_resolutions=baseline_res,
            out_dir=out_dir,
            gallery_basename=OUT_MESH_GALLERY,
            show=args.show,
        )

    print(f"\nSaved: {p1}")
    print(f"Saved: {p2}")
    if p3 is not None:
        print(f"Saved: {p3[0]}")
        print(f"Saved: {p3[1]}")


if __name__ == "__main__":
    main()

