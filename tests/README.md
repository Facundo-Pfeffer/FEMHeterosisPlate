# Test layout

| Directory | Role |
|-----------|------|
| **`unit/`** | Isolated checks: shape functions, material constitutive caching (no full solves). |
| **`integration/`** | Element + mesh + assembly + solver paths: Jacobians/stiffness, patch tests, mesh strategies. |
| **`validation/`** | FE vs classical reference solutions (SSSS / CCCC benchmarks). |

Run everything from the repo root:

```bash
python -m pytest tests/ -q
```

Run a subset, e.g. unit tests only:

```bash
python -m pytest tests/unit/ -q
```
