# 1D Heat Equation Solver (Explicit Finite Difference)

## Overview

This project implements a 1D heat equation solver using an explicit finite difference scheme. It demonstrates numerical methods and validation.

---

## Problem Definition

The solver models transient heat conduction in a 1D rod:

dT/dt = α d²T/dx²

with Dirichlet boundary conditions:

* T(0) = T1
* T(L) = T2

Initial condition:

* T(x,0) = Ti

---

## Numerical Method

* Explicit finite difference scheme
* Central difference in space
* Forward Euler in time

Update equation:

T_i^(n+1) = T_i^n + CFL (T_{i+1}^n - 2T_i^n + T_{i-1}^n)

Stability:

CFL = α dt / dx² ≤ 0.5

Convergence:

R(n) = max|T^n - T^(n-1)| / R(0)

---

## Features

* Input validation with rule-table-based checks
* Loop-based and vectorized solvers
* Normalized residual tracking
* Analytical solution comparison
* Error metrics (L∞, L2, %)
* Solution and residual export (TXT)
* Config-driven execution
* Visualization (profiles, residuals)
* Animated space–time contour
* Solver performance comparison

---

## Validation

Steady-state analytical solution:

T(x) = T1 + (T2 - T1)(x / L)

Validation includes:

* Numerical vs analytical comparison
* L∞ and L2 error norms
* Residual decay

---

## Performance Comparison

Two implementations:

* Loop-based
* Vectorized (NumPy)

---

## Usage

### Configure parameters

Edit `config.py`:

```python
config = {
    "diffusivity": 110,
    "rod_length": 2,
    "nodes": 100,
    "time": 1,
    "t1": 100,
    "t2": 0,
    "ti": 20,
    "target_CFL": 0.4,
    "target_residuals": 1e-3,

    "animate": True,
    "fps": 30,
    "plot": True,
    "export_solution": True,
    "export_residuals": True,
    "compare": True,
    "use_vectorised": True,
}
```

---

### Run

```bash
python main_1D_heat_eqn.py
```

---

## Output

### Plots

* Temperature profile (numerical vs analytical)
* Residual vs time (log scale)
![plots][def2]
* Animated evolution (optional)
![contour][def]

### Files

* solution.txt - x, numerical, analytical
* residuals.txt - iteration, time, residual

---

## Limitations

* Explicit scheme (CFL-limited)
* 1D only
* Dirichlet BCs only
* Not optimized for large-scale problems

---

[def]: contour.png
[def2]: plots.png