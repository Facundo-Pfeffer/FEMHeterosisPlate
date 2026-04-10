# FEMHeterosisPlate

Finite element implementation of a shear-deformable isotropic plate solver based on a heterosis quadrilateral element.

Current element/model choices:
- Q8 interpolation for transverse displacement `w`
- Q9 interpolation for rotations `theta_x`, `theta_y`
- selective integration (`3 x 3` for bending, `2 x 2` for shear)
- sparse global assembly and linear solve

---

## 1) Problem target

The repository is organized to solve and study plate problems with the heterosis element, including:
- plate with centered rectangular cutout (assignment case)
- full rectangular/square benchmark plates
- mesh and load sensitivity checks

---

## 2) Install and run

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Tests are **pytest** modules (`tests/test_*.py`, functions named `test_*`). Run:

```bash
python -m pytest tests/ -q
```

---

## 3) Core formulation implemented

The code implements heterosis plate generalized variables:
- `w`
- `theta_x`, `theta_y`

Element local DOF ordering:

```text
q_e = [
  w1, w2, w3, w4, w5, w6, w7, w8,
  theta_x1, theta_y1,
  theta_x2, theta_y2,
  ...,
  theta_x9, theta_y9
]^T
```

Total local DOFs: `8 + 2*9 = 26`.

Global unknown layout:

```text
u = [w all nodes, theta_x/theta_y all theta nodes]^T
```

So:
- `w(node_id) -> node_id`
- `theta_x(theta_node_id) -> n_w + 2*theta_node_id`
- `theta_y(theta_node_id) -> n_w + 2*theta_node_id + 1`

---

## 4) Repository structure

```text
src/plate_fea/
├── __init__.py
├── assembly.py                 # global K and F assembly
├── boundary_conditions.py      # BC/load dataclasses
├── materials.py                # PlateMaterial, constitutive matrices
├── mesh.py                     # HeterosisMesh data model + node queries
├── mesh_generation.py          # assignment and benchmark mesh generators
├── model.py                    # PlateModel and global dof helpers
├── problem_orchestrator.py     # high-level solve workflows
├── quadrature.py               # Gauss rules (cached)
├── solver.py                   # constrained sparse solve
└── elements/
    ├── base.py                 # element interface
    └── heterosis_plate.py      # shape functions, B-matrices, local K/f
```

Scripts:

```text
scripts/
├── run_smoke_test.py
├── run_problem.py
├── run_patch_test_linear_field.py
├── run_ssss_square_uniform_pressure.py
├── run_clamped_square_uniform_pressure.py
├── plot_mesh.py
├── plot_mesh_sliders.py
└── plot_mesh_demo.py
```

Tests:

```text
tests/
├── test_shape_functions.py
├── test_material_constitutive_cache.py
├── test_patch_linear_field.py
├── test_ssss_uniform_pressure_vs_navier.py
└── test_clamped_square_uniform_pressure.py
```

---

## 5) High-level workflows

### 5.1 Assignment-like solve (plate with cutout)

Use orchestrator entry script:

```bash
python scripts/run_problem.py --resolution 2 --hole-refine 2 --buffer 30
```

Pipeline:
1. mesh generation (`UniformBufferRingQ8Generator`)
2. model/material/element setup
3. boundary conditions
4. line loads
5. assembly (`K`, `F`)
6. constrained solve
7. point-of-interest deflection extraction

### 5.2 Linear closed-form patch test

```bash
python scripts/run_patch_test_linear_field.py
```

Checks exact linear field reproduction with near machine-precision error.

### 5.3 Simply supported square plate under uniform pressure (Navier check)

```bash
python scripts/run_ssss_square_uniform_pressure.py --nx 20 --ny 20
```

SI defaults: 1 m span, 5 mm thickness, 200 GPa, uniform −10 kPa pressure. All edges pinned in translation (`w = 0`); edge moments natural. Compares FE centre deflection [m] to the Kirchhoff–Navier series (shear deformable element is slightly more flexible than thin-plate theory).

### 5.4 Clamped square plate under uniform pressure (Kirchhoff β check)

```bash
python scripts/run_clamped_square_uniform_pressure.py --nx 20 --ny 20
```

Same SI defaults as §5.3; all edges clamped (`w = θ_x = θ_y = 0` on the boundary). Compares FE centre deflection to the classical thin-plate value `w = β q a⁴ / D` with `β ≈ 0.00126532` (tabulated for `ν ≈ 0.3`, e.g. Timoshenko & Woinowsky-Krieger).

---

## 6) Mesh controls

### Static mesh plot

```bash
python scripts/plot_mesh.py --resolution 2 --hole-refine 2 --buffer 30
```

### Interactive slider plot

```bash
python scripts/plot_mesh_sliders.py
```

Sliders:
- `resolution`: global density
- `hole_refine`: extra refinement near hole
- `buffer`: symmetric buffer ring thickness around hole

---

## 7) Implementation notes

- Constitutive matrices are precomputed once per `PlateMaterial` instance and stored read-only.
- Quadrature rules are cached (`lru_cache`) to avoid repeated allocation in element loops.
- Jacobian positivity is checked at quadrature points.
- Global assembly uses sparse CSR matrices.

---

## 8) Current status

Working and tested:
- heterosis element shape functions and mappings
- local stiffness + edge/surface load vectors
- sparse global assembly and constrained solve
- plate-with-hole workflow orchestration
- benchmark/feasibility scripts and automated tests

This is a solid baseline for:
- assignment result generation
- convergence studies
- further validation against FEAP/reference solutions.
