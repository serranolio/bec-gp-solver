# bec-gp-solver

Numerical solver for the Gross-Pitaevskii equation (GPE) describing
N-component Bose-Einstein condensates (BECs). Supports 1D, 2D, 3D Cartesian,
and 3D axially-symmetric geometries. Time evolution uses a 4th-order
Runge-Kutta (RK4) integrator in both real and imaginary time.

## Physics background

A BEC is a state of matter where a dilute gas of bosons occupies the same
quantum ground state below a critical temperature. Its macroscopic wavefunction
ψ is governed by the Gross-Pitaevskii equation, a nonlinear Schrödinger
equation:

```
i ∂ψ/∂t = [-½∇² + U(r) + g|ψ|²] ψ
```

For a spin-orbit coupled N-component system the equation generalises to a
vector equation per component:

```
i ∂ψ_i/∂t = Σ_j [-½∇²δ_ij + (σ_z)_ij(-i∂/∂z) + Ω_ij/2 + E_ij] ψ_j
           + Σ_j g_ij n_j ψ_i  +  U ψ_i
```

where `σ_z` encodes the spin-orbit coupling, `Ω` is the Rabi coupling matrix,
`E_diag` contains the single-particle energies (detuning + Zeeman), and `g_ij`
is the N×N interaction matrix.

Ground states are found via imaginary-time evolution, which replaces `i∂/∂t`
with `∂/∂t` and subtracts the chemical potential μ = ⟨H⟩ at each step to
prevent the wavefunction from decaying to zero.

## Project structure

```
bec-gp-solver/
├── src/
│   └── bec_gp_solver/
│       ├── geometry.py       # Grid, volume elements, kinetic operators
│       ├── gp_equation.py    # RHS of the GP equation for N components
│       └── integrator.py     # RK4 solver
├── tests/
│   ├── test_geometry.py
│   └── test_gp_equation.py
├── configs/                  # TOML files, one per simulation run
│   └── example.toml
├── run_simulation.py         # Entry point
├── pyproject.toml
└── pixi.toml
```

## Installation

The project uses [pixi](https://prefix.dev) for environment management.

```bash
git clone https://github.com/serranolio/bec-gp-solver
cd bec-gp-solver
pixi install
```

To make the package importable in editable mode (needed for the tests):

```bash
pip install -e .
```

## Usage

### 1. Define a config file

Copy and edit one of the examples in `configs/`:

```toml
# configs/my_run.toml

[geometry]
kind    = "3d_axial"
nx      = 64
nz      = 512
lx      = 15.0
lz      = 60.0

[physics]
n_atoms    = 193369
omega_r    = 2.7
detuning   = 0.0
q_zeeman   = 7.189
n_components = 3
sigma_z_diag = [-2.0, 2.0, 6.0]

[numerics]
steps_groundstate = 5000
step_size         = 0.01
frames            = 100
output_dir        = "output/my_run"
```

### 2. Run the simulation

```bash
python run_simulation.py --config configs/my_run.toml
```

Results are saved as `.npy` files in the directory specified by `output_dir`.

### 3. Use the modules directly

```python
import numpy as np
from bec_gp_solver.geometry import make_geometry
from bec_gp_solver.gp_equation import build_params, rhs
from bec_gp_solver.integrator import rk4

# build geometry (no physics here)
geo = make_geometry('3d_axial', nx=64, nz=512, lx=15.0, lz=60.0)

# build trap potential using the grid coordinates
r_, z_ = geo.grids
U = (wx * r_)**2 / 4 + (wz * z_)**2 / 4

# assemble params dict
params = build_params(
    n_components = 3,
    omega_r      = 2.7,
    detuning     = 0.0,
    q_zeeman     = 7.189,
    sigma_z_diag = [-2.0, 2.0, 6.0],
    g_matrix     = g * np.ones((3, 3)),
    n_atoms      = 1.93369e5,
    U            = U.reshape(-1),
    geo          = geo,
)

# find ground state via imaginary-time evolution
psi0 = ...   # initial guess, shape (3 * nx * nz,)

psi_gs = rk4(
    fun       = rhs,
    y0        = psi0,
    frames    = 2,
    steps     = 5000,
    step_size = 0.01,
    geo       = geo,
    params    = params,
    mode      = 'imaginary',
)[:, -1]

# real-time evolution
psi_t = rk4(
    fun       = rhs,
    y0        = psi_gs,
    frames    = 100,
    steps     = 10000,
    step_size = 0.005,
    geo       = geo,
    params    = params,
    mode      = 'real',
)
```

### 4. Time-dependent parameters

Both `detuning` and `lattice` in the params dict accept callables for ramps:

```python
# linear ramp of the lattice from 0 to omega_max over time T_ramp
params['lattice'] = lambda t: (t / T_ramp) * omega_max * lattice_profile
```

## Running the tests

```bash
pytest tests/ -v
```

## Units

All quantities are in recoil units:

| Quantity | Unit |
|----------|------|
| Energy   | `ℏ × 2π × f_recoil` |
| Length   | `ℏ / sqrt(2 m E_unit)` |
| Time     | `ℏ / E_unit` |
