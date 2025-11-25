# Macrospin LLG (Python)

A Python package for solving the macrospin Landau-Lifshitz-Gilbert (LLG) equations for an arbitrary number of magnetic moments, including the effects of damping, Dzyaloshinskii–Moriya interaction (DMI), and arbitrary effective fields.

## Features

- Arbitrary number of macrospins.
- Supports exchange, multiple anisotropies, DMI, Zeeman field, and time-dependent fields.
- Robust, explicit LLG solver (scipy backend) with optional implicit solver (Assimulo/IDA backend).
- Interactive visualization using Plotly.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Zeleznyj/macrospin_llg_python
   ```
2. Install dependencies (see `pyproject.toml`).
   ```bash
   pip install -e .
   ```

---

## Usage

### Defining a Model

```python
from macrospin_llg import Model

n_mag = 3  # Number of magnetic moments
m = Model(n_mag)
```

#### Adding Interactions

- `m.add_exchange(a, b, J)` - Symmetric exchange between moments `a` and `b`.
- `m.add_ani2(a, K, A)`     - Uniaxial anisotropy (axis `A`).
- `m.add_ani4(a, K)`        - Cubic anisotropy.
- `m.add_ani6(a, K, A)`     - Sixth-order anisotropy (axis `A`).
- `m.add_DMI(a, b, d)`      - Antisymmetric DMI interaction (vector `d`).
- `m.add_B(a, B)`           - Static magnetic field (B as 3-vector).
- `m.add_custom_interaction(name, beff_per_atom=None, energy=None)` - Named custom terms:
  - `name`: unique string identifier for this term.
  - `beff_per_atom`: sequence length `n_mag` with per-atom callables `beff_i(t, M) -> (3,)` (use `None` to skip a site).
  - `energy(t, M) -> float` contributes to total energy (optional). Provide both for conservative interactions.
  - Call multiple times to add multiple interactions; all fields and energies sum, tracked by name.

Arguments `a` and `b`:
- Integer (zero-based): Specifies one moment. Indexing starts from 0, unlike in the Matlab code!
- -1: All moments.
- List: Multiple moments.

#### Damping

```python
m.ag = 0.01  # Set Gilbert damping
```

#### Solving the Model

```python
M0 = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1]
])  # Shape (n_mag, 3)

solution = m.solve_LLG(tf=1.0, M0=M0)
```
Additional solver parameters: `t0`, `method`, `t_eval`, and any keyword arguments accepted by `scipy.integrate.solve_ivp`.

You can configure solver options by passing them as keyword arguments:

```python
# Example with custom solver settings
solution = m.solve_LLG(
    tf=1.0, 
    M0=M0, 
    method='DOP853',  # Default method (Runge-Kutta 8(5,3))
    rtol=1e-6,        # Relative tolerance
    atol=1e-9,        # Absolute tolerance
    t_eval=np.linspace(0, 1.0, 200)  # Specific time points for output
)
```

### Solver backend

The default solver uses **scipy's explicit ODE solver** via `scipy.integrate.solve_ivp`:

- <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html" target="_blank" rel="noopener">scipy.integrate.solve_ivp documentation</a>

The LLG equations are transformed to explicit form and solved using Runge-Kutta methods (default: `DOP853`). Any parameter supported by `solve_ivp` can be passed as a keyword argument to `solve_LLG` (e.g., `method`, `rtol`, `atol`, `max_step`, `dense_output`).

#### Alternative: Implicit solver

For stiff problems or specific use cases, an **implicit DAE solver** is available via `solve_LLG_implicit`:

```python
solution = m.solve_LLG_implicit(
    tf=1.0, 
    M0=M0, 
    ncp=200,  # Number of communication points
    solver_kwargs={'rtol': 1e-6, 'atol': 1e-9}
)
```

This uses the IDA solver via the Assimulo interface:
- <a href="https://jmodelica.org/assimulo/DAE_IDA.html" target="_blank" rel="noopener">Assimulo documentation</a>
- <a href="https://sundials.readthedocs.io/en/latest/ida/index.html" target="_blank" rel="noopener">SUNDIALS IDA documentation</a>

Note: The implicit solver requires the optional `assimulo` package. Install it with `pip install macrospin_llg[assimulo]`.

---

## Solution Object

After running `solve_LLG`, you get a `Solution` object with:

- `solution.t`         - Array of time points.
- `solution.M`         - Array of magnetic moments, shape `(time_points, n_mag, 3)`.
- `solution.E`         - Energy evolution (computed on first access or via plotting).
- `solution.M_total`   - Net magnetic moment as a function of time (computed on first access).

### Plotting

```python
solution.plot()              # Plots Mx, My, Mz for each moment
solution.plot_M_total()      # Plots net magnetic moment (Mx, My, Mz)
solution.plot_energy()       # Plots energy breakdown from dict: Total and each named component

# Animated 3D display (optional)
solution.plot_animated_3d()
```

---

## Model Data Structures

You can access model parameters directly, for advanced use:

- `m.J`    - Exchange tensor, shape `(3, 3, n_mag, n_mag)`
- `m.D`    - DMI tensor, shape `(3, 3, n_mag, n_mag)`
- `m.K2`, `m.K4`, `m.K6`      - Anisotropy constants per moment
- `m.A2`, `m.A6`              - Anisotropy axes, shape `(3, n_mag)`
- `m.B`    - Fields, shape `(n_mag, 3)`
- `m.ag`   - Damping
- `m._custom_beff_terms` - Dicts of custom effective field functions per moment (internal API).
- `m._custom_energy_terms` - Dict of custom energy functions by name (internal API).

---

## Units

- Magnetic moments: Bohr magneton (\(\mu_B\))
- Magnetic fields: Tesla (T)
- Damping: dimensionless
- Anisotropy constants: eV
- Time: nanoseconds (ns) in plotting, seconds in computations

---

## Hamiltonian and LLG Equations

### Hamiltonian
The system Hamiltonian is

$$
H = H_{\text{ex}} + H_{\text{DMI}} + H_{\text{ani}} + H_B
$$

where

- **Exchange:**
  
  $$H_{\text{ex}} = \sum_{ab} \frac{J_{ij}^{ab}}{2} \hat{M}_i^a \hat{M}_j^b$$
  
- **DMI:**
  
  $$H_{\text{DMI}} = \sum_{ab} \frac{D_{ij}^{ab}}{2} \hat{M}_i^a \hat{M}_j^b$$
  
  with $D_{jk} = d_i \epsilon_{ijk}$ ($\epsilon_{ijk}$ is the Levi-Civita symbol).
- **Anisotropy:**
  
  $$H_{\text{ani}} = \sum_{a} -\frac{K_2^a}{2}(\hat{M}^a \cdot \hat{A}_2^a)^2-
  \frac{K_4^a}{2}(\sum_i \hat{M}_i^4)\\
  -\frac{K_6^a}{2}\left((\hat{M}^a_x )^6 - (\hat{M}^a_y )^6 - 15(\hat{M}^a_x )^4(\hat{M}^a_y )^2 + 15(\hat{M}^a_x )^2(\hat{M}^a_y )^4\right)$$
  
  Note that the 6-fold anisotropy corresponds to an anisotropy in the xy plane with the form $\cos(\theta)$, where $\theta$ is the in-plane angle measured from the x-axis.
- **Zeeman (field):**
  
  $$H_B = \sum_{a} \hat{M}^a \cdot B^a$$
  

The sums run over all sites $a, b$ (with $a \neq b$), and $\hat{M}^a = M^a/|M^a|$. $J$ is symmetric while $D$ is antisymmetric,
see code for conventions.

### LLG Equations

The Landau-Lifshitz-Gilbert equations in implicit form are:

$$
\frac{d M^a}{dt} = -\gamma M^a \times B_{\text{eff}}^a + \frac{\alpha}{|M^a|} M^a \times \frac{d M^a}{dt}
$$

For the default explicit solver, these are transformed to explicit form:

$$
\frac{d M^a}{dt} = -\frac{\gamma}{1 + \alpha^2} M^a \times B_{\text{eff}}^a - \frac{\gamma \alpha}{(1 + \alpha^2)|M^a|} M^a \times (M^a \times B_{\text{eff}}^a)
$$

where

$$
B_{\text{eff}}^a = -\frac{1}{|M^a|}\frac{\partial H}{\partial \hat{M}^a}
$$

- $\alpha$ is the Gilbert damping (set as `m.ag`)
- $\gamma > 0$ is the gyromagnetic ratio

All terms admitted by the Hamiltonian and equation above are supported by corresponding `Model` class functions.

---

## License

This project is licensed under the Mozilla Public License Version 2.0 (MPL-2.0).
See the `LICENSE` file for full text.

---

## References

- Based on original [MATLAB macrospin-llg project](https://bitbucket.org/zeleznyj/macrospin-llg) by J. Železný.

---
