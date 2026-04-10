"""
Run the plate-with-hole FEM problem using the high-level orchestrator.

Example:
    python scripts/run_problem.py --resolution 3 --hole-refine 2 --buffer 30
"""

from __future__ import annotations

import argparse

from plate_fea.problem_orchestrator import ProblemConfig, solve_plate_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve plate-with-hole problem with orchestrated FEM steps.")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--hole-refine", type=int, default=2)
    parser.add_argument("--buffer", type=float, default=30.0)
    parser.add_argument("--young-modulus", type=float, default=200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.25)
    parser.add_argument("--thickness", type=float, default=20.0)
    parser.add_argument("--hole-top-shear-load", type=float, default=-1.0)
    parser.add_argument(
        "--clamped-outer-edges",
        type=str,
        default="left,top",
        help="Comma-separated subset of: left,right,bottom,top",
    )
    args = parser.parse_args()

    clamped_edges = tuple(edge.strip() for edge in args.clamped_outer_edges.split(",") if edge.strip())
    config = ProblemConfig(
        resolution=int(args.resolution),
        hole_refine=int(args.hole_refine),
        buffer=float(args.buffer),
        young_modulus=float(args.young_modulus),
        poisson_ratio=float(args.poisson_ratio),
        thickness=float(args.thickness),
        hole_top_shear_load=float(args.hole_top_shear_load),
        clamped_outer_edges=clamped_edges,  # type: ignore[arg-type]
    )
    result = solve_plate_problem(config)

    print("=== Plate with Hole FEM Result ===")
    print(f"elements:           {result.model.mesh.total_element_number}")
    print(f"w nodes:            {result.model.mesh.total_w_node_number}")
    print(f"theta nodes:        {result.model.mesh.total_theta_node_number}")
    print(f"total dof:          {result.model.mesh.total_dof_number}")
    print(f"point A node id:    {result.point_a_node_id}")
    print(f"point A deflection: {result.point_a_deflection:.8e}")


if __name__ == "__main__":
    main()

