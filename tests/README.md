# Test layout

Tests are split into four folders by scope. Run everything from the repo root:

```bash
python -m pytest tests/ -q
```

## Folders

| Folder | Mark | Role |
|--------|------|------|
| `unit/` | *(none)* | Fast, isolated component tests — shape functions, material matrices, element Jacobian/stiffness, mesh structure. No full FEM solve. |
| `patch/` | `patch` | Element patch tests: exact strain recovery and eigenvalue spectrum on distorted meshes. |
| `benchmarks/` | `benchmark` | Comparison against classical Kirchhoff analytical solutions and Hughes textbook results. |
| `final_project_specific/` | `final_project_specific` | Mechanical validation of the CE 222 plate-with-hole assignment: force assembly, reaction equilibrium, moment equilibrium. |

## Running by category

```bash
python -m pytest tests/unit/ -q
python -m pytest -m patch -q
python -m pytest -m benchmark -q
python -m pytest -m final_project_specific -q
```

## Patch tests (`tests/patch/`)

| Test file | What it checks |
|-----------|---------------|
| `test_patch_linear_field.py` | Closed-form linear-field solution recovered by the full solve pipeline. |
| `test_simple_patch_cases.py` | Constant shear field (sampled at 2×2 shear points) and constant κ field (sampled at 3×3 bending points) on the distorted 5-element patch. |
| `test_five_element_patch.py` | All five strain components (κ_xx, κ_yy, κ_xy, γ_xz, γ_yz) match the representable linear kinematic field to 1e-10 on the distorted enclosing patch. |
| `test_single_element_eigen.py` | Distorted single element has exactly 3 near-zero eigenvalues (rigid body modes). |

## Benchmark tests (`tests/benchmarks/`)

| Test file | Reference |
|-----------|-----------|
| `test_ssss_uniform_pressure_vs_navier.py` | Kirchhoff–Navier series for SSSS square plate, uniform pressure. |
| `test_clamped_square_uniform_pressure.py` | Kirchhoff β factor (Timoshenko) for CCCC square plate, uniform pressure. |
| `test_circular_plate_hughes.py` | Hughes & Cohen (1978) Fig. 5.3.19: quarter-circle convergence, SS₁-U and CL-U cases, Reissner-Mindlin reference. |

## Final-project tests (`tests/final_project_specific/`)

| Test | Check |
|------|-------|
| `test_total_applied_force` | `sum(F_w) = q × hole_width = −250 000 N`. |
| `test_reaction_forces_balance_applied_load` | `sum(R_w) = +250 000 N` (reactions at clamped w-DOFs). |
| `test_moment_equilibrium` | Moment balance about x and y axes, including θ-DOF couple reactions. |
