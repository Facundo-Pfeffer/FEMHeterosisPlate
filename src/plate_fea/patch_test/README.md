# Patch testing for the heterosis plate element

This package runs **classical patch-test variants** (often labelled **Test A**, **Test B**, and **Test C** in the literature) on the production **heterosis** quadrilateral: **Serendipity Q8** shape functions for the transverse deflection `w`, and **Lagrange Q9** for the rotations `θ_x`, `θ_y`. The implementation is **plate-specific**: unknowns, strain measures, and “exact” discrete states are those used in `plate_fea`, not a plane-stress `(u, v)` elasticity example.

## Background

The **patch test** and heterosis plate kinematics are standard material. Labels **Test A**, **Test B**, and **Test C** follow **Chapter 11** of **References** [1] (O. C. Zienkiewicz, R. L. Taylor, and S. Govindjee, *The Finite Element Method: Its Basis and Fundamentals*, 8th ed., 2024). **Operational definitions** here are in the next section and in `patch_test/engine.py` (CI and `scripts/run_patch_test_chapter11.py`).

## Problem and notation

### Continuous heterosis plate (context)

On a mid-surface region `Ω ⊂ ℝ²` with coordinates `(x, y)`, the primary variables are the **transverse deflection** `w(x, y)` and **rotations** `θ_x(x, y)`, `θ_y(x, y)` about the `y`- and `x`-axes. Curvatures and transverse shear strains used in the **discrete** heterosis plate formulation are:

| Symbol | Definition |
|--------|------------|
| `κ_xx` | `∂θ_x/∂x` |
| `κ_yy` | `∂θ_y/∂y` |
| `κ_xy` | `∂θ_x/∂y + ∂θ_y/∂x` |
| `γ_xz` | `∂w/∂x − θ_x` |
| `γ_yz` | `∂w/∂y − θ_y` |

A **stress-free, homogeneous** (no distributed pressure `q`) state with **vanishing bending and shear** in the sense above requires `κ = 0` and `γ = 0`, hence **constant** `θ_x`, `θ_y` and **`θ = ∇w`**, so **`w` is affine** in `(x, y)`:

```text
w(x, y) = a x + b y + c ,   θ_x = a ,   θ_y = b .
```

That is a **three-parameter** linear space (modes in `(a, b, c)`). The harness uses an elementary basis of three states; see `plate_exact_states.py`.

### Discrete system used in code

| Symbol | Meaning |
|--------|---------|
| **`ũ`** | Global nodal **displacement vector** (length = total DOFs). Ordering: all `w` DOFs first (at `Q8` nodes), then `θ_x` and `θ_y` at `Q9` θ-nodes via `PlateModel.get_theta_x_dof` / `get_theta_y_dof`. Built for patch fields by `nodal_vector.py`. |
| **`K`** | Global **tangent stiffness** from `assemble_stiffness_matrix(model)` (sparse). |
| **`f`** | Global **force vector** from `assemble_force_vector(model)`; **`f = 0`** for the homogeneous linear patch runs. |
| **Essential BCs** | Prescribed components of `ũ` at selected nodes (`EssentialBoundaryCondition`); assembled into index array **`bc_ess`** and values **`bc_val`** by `model.build_essential_boundary_arrays()`. |
| **`K_ff`, `f_f`** | Blocks on **free** (unprescribed) DOFs after partitioning; used for stability diagnostics. |
| **Residual (equilibrium check)** | `‖ K ũ − f ‖` (Euclidean norm in code; see `diagnostics.residual_norm`). |

There is **no** second “shadow” stiffness: the patch harness calls the same assembly and solver as production (`plate_fea/solver.solve_linear_system`).

In continuous notation the boundary-value problem is often written **`A(u) = 0`** on `Ω` with **`B(u) = 0`** on `∂Ω`, with **`u = (w, θ_x, θ_y)`** and a finite-element approximation **`u_h = N ũ`**. In the table above, **`ũ`** means only the global nodal vector assembled by this library.

## Tests A, B, and C

All three tests use the **same** linear exact nodal vector **`ũ_exact`** from `assemble_linear_patch_displacement` for a given `LinearHeterosisPatchState`, **`f = 0`**, and the same `K` from the mesh + material + quadrature preset.

### Test A — consistency / “reaction” test

- **Boundary / loads:** none prescribed; model has **no** essential BCs and **`f = 0`**.
- **Check:** evaluate **`r = ‖ K ũ_exact − f ‖ = ‖ K ũ_exact ‖`**. Pass if **`r`** is below `residual_tol` (`PatchTestSuiteConfig`).
- **Meaning:** the linear patch nodal pattern is a **null load** equilibrium mode of the **discrete** system (consistency of `K` with the chosen `ũ` for that polynomial state). It does **not**, by itself, prove stability or correct boundary coupling.

### Test B — boundary prescription, interior solve

- **Boundary:** **all** nodes on the **outer rectangle** that carry **`w`** or **`θ`** in this mesh get essential values from the linear state (`engine._apply_boundary_linear_state`).
- **Check:** solve for the remaining DOFs; compare **`ũ`** to **`ũ_exact`** (max-norm and L2 on the full vector); require small equilibrium residual and stable **`K_ff`** diagnostics.
- **Meaning:** reproduces the exact linear field when the patch boundary is fully clamped to that field (patch of elements “immersed” in the correct hard boundary data).

### Test C — minimal essential data + traction-free remainder (primary regression)

- **Boundary:** **one** corner node: prescribe **`w`, `θ_x`, `θ_y`** only there (`engine._apply_minimal_corner_linear`) — enough to remove the three rigid / linear null modes of the patch state. All other boundaries are **natural** (no extra loads in `f`; traction-free in the weak form sense implemented by the model).
- **Check:** same as Test B (nodal error, residual, **`K_ff`** rank / spectrum heuristics, optional load perturbation sensitivity).
- **Meaning:** closest to the **standard patch test** used to catch **rank deficiency**, spurious mechanisms, or locking on a **multi-element** mesh. **`acceptance_all_pass`** in `engine.py` treats **Test C on a regular multi-element patch with standard quadrature** as the **decisive CI gate**; a **single-element** pass alone is **never** treated as sufficient.

---

## Quadrature presets (`quadrature_presets`)

Labels refer to tensor-product Gauss rules passed through `PlateModel.element_stiffness_kwargs` into the same element routine as production:

| Label | Bending rule | Shear rule |
|-------|----------------|------------|
| `standard (3×3 bend + 2×2 shear)` | `(3,3)` | `(2,2)` |
| `reduced bending …` | `(2,2)` | `(2,2)` |
| `reduced shear …` | `(3,3)` | `(1,1)` |
| `reduced both …` | `(2,2)` | `(1,1)` |

---

## Vocabulary: “bubbles” and incompatible modes

In **mixed or enhanced** elements, authors sometimes add **internal** shape functions (often written in parent coordinates, e.g. factors like `1 − ξ²`) that are **non-nodal**. Those are loosely called **bubble** or **incompatible** modes, and patch tests may partition stiffness as **`[K_uu, K_ua; …]`** with internal parameters **`a`**.

**This element does not add extra internal DOFs:** only `Q8` geometry for `w` and `Q9` for rotations. There is **no** internal bubble block to test beyond the nodal fields.

---

## Relation to the production solver

The harness uses the same calls as `solve_displacement_system` in `plate_fea/solver.py`:

- `assemble_stiffness_matrix(model)`
- `assemble_force_vector(model)`
- `model.build_essential_boundary_arrays()`
- `solve_linear_system(K, F, bc_ess, bc_val)`

What differs from a full problem driver is **mesh topology**, **BC/load setup**, and **orchestration** — not a duplicate `K` or solver.

---

## Package layout

| Module | Purpose |
|--------|---------|
| `engine.py` | Tests A/B/C, suite config, `acceptance_all_pass`, summary text |
| `plate_exact_states.py` | Linear patch state; quadratic `w` with `θ = ∇w` (γ = 0) for manufactured checks |
| `nodal_vector.py` | Map exact fields to global `ũ` |
| `geometry.py` | Single- vs multi-element patches; distorted variant |
| `diagnostics.py` | Residual, rank / spectrum of `K_ff`, perturbation probe |
| `manufactured.py` | `f = K ũ` for consistency without analytical `q` |
| `reporting.py` | Structured pass/fail and failure class |

---

## Running

From the repository root (after `pip install -e .`):

```bash
python scripts/run_patch_test_chapter11.py
```

Flags: `--no-distorted`, `--no-single`, `--no-multi`, `--one-linear-mode`, `--no-reduced-quadrature`.

Pytest (fast subset):

```bash
python -m pytest tests/integration/test_patch_chapter11_framework.py -q
```

Element Jacobian / stiffness checks (symmetry, eigenvalue count, positive `det J` at Gauss points, global `K` symmetry):

```bash
python -m pytest tests/integration/test_element_jacobian_and_stiffness.py -q
```

---

## Implemented scope

- **Linear** exact states: `w = a x + b y + c`, `θ_x = a`, `θ_y = b` (vanishing bending and transverse shear in the **discrete `B`-matrix** sense used in code).
- **Test A / B / C** as defined above; **single-element** and **multi-element** patches; optional **distorted** interior nodes.
- **Quadrature sweeps** via `element_stiffness_kwargs`.
- **Manufactured higher-order:** build `f_mfg := K ũ` for a chosen discrete `ũ` (e.g. quadratic `w` with `θ = ∇w`); Test A–style residual vanishes by construction — checks **assembly consistency**, not analytical equilibrium of the PDE for arbitrary polynomials without deriving `q(x, y)`.

## Out of scope (documented limits)

- General **analytical** `q(x, y)` and edge resultants for **quadratic+** equilibrium of the heterosis strong problem.
- Dedicated **stress/strain at Gauss points vs nodal** audit.
- Full **weak-patch** theory for mapped/curvilinear elements (distorted meshes are **empirical** checks only).
- **Block** patch tests for incompatible internal variables (**none** in the current element).

---

## Acceptance rule (CI)

`engine.acceptance_all_pass`: **every** report with **`test_type == "C"`**, **`patch_topology == "multi_element"`**, **standard quadrature**, and **regular** (non-distorted) geometry must pass. Default tolerances are `PatchTestSuiteConfig.nodal_tol` and `residual_tol` in `engine.py`.

## References

1. O. C. Zienkiewicz, R. L. Taylor, and S. Govindjee, *The Finite Element Method: Its Basis and Fundamentals*, 8th ed., Butterworth-Heinemann, Elsevier, 2024. ISBN 978-0-443-16044-8. **Chapter 11** in this volume covers patch tests and related ideas in the spirit of the harness here.
