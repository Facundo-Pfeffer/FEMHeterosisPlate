"""
Patch-test engine for the heterosis plate (Tests A/B/C, diagnostics, quadrature variants).

Terminology follows Ch. 11 of Zienkiewicz, Taylor & Govindjee, *The Finite Element Method:
Its Basis and Fundamentals* (8th ed.). See **``README.md``** in this package for notation,
vocabulary, scope, and how to run the suite.
"""

from plate_fea.patch_test.engine import PatchTestSuiteConfig, run_patch_test_suite, summarize
from plate_fea.patch_test.plate_exact_states import LinearHeterosisPatchState
from plate_fea.patch_test.reporting import FailureClass, PatchTestCaseReport

__all__ = [
    "FailureClass",
    "LinearHeterosisPatchState",
    "PatchTestCaseReport",
    "PatchTestSuiteConfig",
    "run_patch_test_suite",
    "summarize",
]
