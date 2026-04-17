#!/usr/bin/env python3
"""Run the heterosis plate patch-test suite (Tests A/B/C; Ch. 11, Zienkiewicz et al., *FEM: Basis and Fundamentals*, 8th ed.) and print reports."""

from __future__ import annotations

import argparse

from plate_fea.patch_test.engine import PatchTestSuiteConfig, run_patch_test_suite, summarize


def main() -> None:
    p = argparse.ArgumentParser(description="Plate heterosis patch tests (Tests A, B, C).")
    p.add_argument("--no-distorted", action="store_true", help="Skip distorted multi-element patch.")
    p.add_argument("--no-single", action="store_true", help="Skip single-element patches.")
    p.add_argument("--no-multi", action="store_true", help="Skip regular multi-element patch.")
    p.add_argument("--one-linear-mode", action="store_true", help="Use a single (a,b,c) instead of three.")
    p.add_argument("--no-reduced-quadrature", action="store_true", help="Only standard integration rule.")
    args = p.parse_args()

    cfg = PatchTestSuiteConfig(
        run_distorted=not args.no_distorted,
        run_single_element=not args.no_single,
        run_multi_element=not args.no_multi,
        test_all_linear_modes=not args.one_linear_mode,
        full_quadrature_sweep=not args.no_reduced_quadrature,
    )
    reports = run_patch_test_suite(cfg)
    print(summarize(reports))


if __name__ == "__main__":
    main()
