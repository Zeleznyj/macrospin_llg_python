"""
Chart-free numeric Gamma-point (q = 0) magnon modes of an arbitrary macrospin
model.

Unlike an implementation built on spherical angles (theta, phi), this module
never divides by sin(theta): each site gets its own orthonormal tangent frame
constructed from a retraction onto the unit sphere, so the result is regular
everywhere, including equilibria where a moment points along the global z
axis. It differentiates the model's own `energy` and `Beff` numerically, so
no model-specific analytic Hessian is required.

Only three things are used from the caller's model object:

    model.energy(t, M)["total"]   -> float,        eV
    model.Beff(t, n, M)            -> ndarray (3,), Tesla
    Model.mu_B, Model.gamma_e      -> class attributes (optional)

`M` is `(n_mag, 3)`, Cartesian moments carrying their magnitude in units of
the Bohr magneton (not unit vectors); `Beff` returns the field at site `n`
given the whole configuration.

**Static energy only.** `energy` and `Beff` accept a time argument `t` for
models with time-dependent drives, but linearising about an equilibrium is
only meaningful when the energy is static at that instant. `t` is accepted
here and passed through to the model unchanged; the caller is responsible
for supplying a `t` at which the energy does not change in time (e.g. a
constant field, or a drive evaluated at the instant of interest treated as
frozen). This module does not attempt to handle genuinely time-dependent
drives -- it linearises `E(t=t0, .)` as if it were static.

Gilbert damping `alpha` and the gyromagnetic ratio `gamma` used in the
symplectic prefactor are arguments to `calculate_modes()`, not read off the
model instance: `Beff` is the field conjugate to the energy alone and carries no
damping, and the caller may want undamped modes regardless of any damping
the model uses elsewhere for time integration. `gamma` defaults to the
model's `gamma_e` class attribute (or the module fallback) if not given
explicitly.

Pitfalls
--------
- **Goldstone / zero modes.** A continuous symmetry (e.g. rotation about the
  field axis at zero anisotropy) gives a degenerate zero eigenvalue pair of
  `W @ H` that is generically *defective* -- a 2x2 Jordan block, not a
  genuine two-dimensional eigenspace. Its motion is a linear drift in time,
  not a periodic orbit, so any trajectory or ellipse built from it is
  meaningless. `calculate_modes()` flags this by counting `H`'s eigenvalues
  that are near zero relative to its largest one (tolerance `soft_tol`,
  returned as `n_soft`) and warning if that count is odd -- `matrix_rank(W @
  H)` cannot be used for this because `W`'s `gamma/mu_eV` prefactor (~1e15)
  swamps the rank-revealing threshold `matrix_rank` derives from `D`'s
  singular values, so it never sees the deficiency. See
  `calculate_modes()`'s docstring for the expected `n_soft` in the standard
  two-macrospin, zero-field cases.
- **Saddle points print as omega = 0, not as an error.** A saddle point of
  `H` gives `W @ H` a purely real eigenvalue pair -- exponential growth, not
  oscillation -- and since `omega = -Im(eigenvalue)` that branch silently
  prints as `omega = 0` unless you check `growth_rate`/`h_eigvals` too.
  `calculate_modes()` warns and reports `growth_rate` for exactly this
  reason; it also distinguishes a saddle (a negative `h_eigvals[0]` beyond
  `soft_tol`) from a genuinely gapless mode (an odd count of near-zero
  eigenvalues with nothing negative) in the warning text, since the two call
  for different fixes -- re-relax from a different configuration for the
  former, recheck your expected symmetry count for the latter.
- **Complex eigenvectors.** The motion is precessional, so eigenvectors are
  complex by nature. The physical trajectory is `Re[v * exp(-i*omega*t)]`;
  taking `.real` on its own gives only a `t = 0` snapshot, not the mode.
- **Linear regime.** If a trajectory helper is built on top of this, the
  linearisation is only valid for tangent amplitudes of order 0.01-0.03 rad;
  the relative error against the full nonlinear LLG grows roughly as
  `3 * amplitude`.
- **Step size.** The default `h=1e-4` assumes a model energy accurate to
  near machine precision. Any internal tolerance in the model (Ewald sums,
  iterative solvers, ...) raises the finite-difference noise floor and moves
  the optimal step up; run an h-scan once on your own model (see the
  "grad" method and acceptance test 5 below) and read the minimum off it.
"""

from __future__ import annotations

import warnings

import numpy as np

MU_B_EV_PER_T = 5.7883818012e-5   # Bohr magneton [eV/T], fallback only
GAMMA_RAD_S_T = 1.760859644e11    # free-electron gyromagnetic ratio [rad/(s T)], fallback only


def _get_constants(model):
    """Read (mu_B, gamma_e) off the model class/instance, else fall back."""
    mu_B = getattr(model, "mu_B", MU_B_EV_PER_T)
    gamma_e = getattr(model, "gamma_e", GAMMA_RAD_S_T)
    return float(mu_B), float(gamma_e)


# ---------------------------------------------------------------------------
# tangent frames and retraction
# ---------------------------------------------------------------------------
def tangent_frames(m_hat):
    """Build an orthonormal tangent frame (e_i1, e_i2) at every site.

    `e_i1 x e_i2 = m_hat_i` for each site `i`. `e_i1` is built by crossing
    `m_hat_i` with whichever global Cartesian axis has the smallest
    `|component|` of `m_hat_i` (never degenerate, since the three components
    cannot all simultaneously be within a hair of 1), then `e_i2 = m_hat_i x
    e_i1`.

    Parameters
    ----------
    m_hat : (n_mag, 3) unit vectors

    Returns
    -------
    frames : (n_mag, 2, 3), frames[i, 0] = e_i1, frames[i, 1] = e_i2
    """
    m_hat = np.asarray(m_hat, float)
    n_mag = m_hat.shape[0]
    frames = np.empty((n_mag, 2, 3))
    axes = np.eye(3)
    for i in range(n_mag):
        mh = m_hat[i]
        g = axes[np.argmin(np.abs(mh))]
        e1 = np.cross(mh, g)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(mh, e1)
        assert np.dot(np.cross(e1, e2), mh) > 0.999, (
            f"left-handed tangent frame at site {i}; this reverses that "
            "site's precession sense and silently corrupts the spectrum"
        )
        frames[i, 0] = e1
        frames[i, 1] = e2
    return frames


def retract(x, m_hat0, frames):
    """Retract flat tangent coordinates `x` back onto the unit sphere.

        u_i(x)     = m_hat0_i + x_i1 * e_i1 + x_i2 * e_i2
        r_i(x)     = |u_i(x)|
        m_hat_i(x) = u_i(x) / r_i(x)

    Parameters
    ----------
    x       : (2*n_mag,) flat, [x_01, x_02, x_11, x_12, ...]
    m_hat0  : (n_mag, 3) equilibrium unit vectors
    frames  : (n_mag, 2, 3) from `tangent_frames`

    Returns
    -------
    m_hat : (n_mag, 3) retracted unit vectors
    r     : (n_mag,) retraction radii (never called with M off-sphere: see
            `grad_tangent`, which divides by `|M_i|` again inside `Beff`)
    """
    x = np.asarray(x, float).reshape(-1, 2)
    u = m_hat0 + x[:, 0:1] * frames[:, 0, :] + x[:, 1:2] * frames[:, 1, :]
    r = np.linalg.norm(u, axis=1)
    m_hat = u / r[:, None]
    return m_hat, r


def grad_tangent(model, x, m_hat0, frames, target_norms, mu_eV, t=0.0):
    """Gradient of the retracted energy `E_tilde(x) = E(M(x))` in tangent
    coordinates.

        dE_tilde/dx_ia = e_ia . P_i(x) G_i(x) / r_i(x)

    with `P_i(x) = 1 - m_hat_i(x) m_hat_i(x)^T` the tangent-plane projector
    at the *displaced* point and `G_i(x) = dE/dm_hat_i = -mu_eV_i *
    Beff(t, i, M(x))`. Both the projection and the `1/r_i` factor are
    required: omitting either gives a gradient that is correct only at
    `x = 0` and wrong in its neighbourhood, which silently corrupts any
    Hessian obtained by differencing this function.

    Parameters
    ----------
    model         : object exposing `Beff(t, n, M) -> (3,)` in Tesla
    x             : (2*n_mag,) flat tangent coordinates
    m_hat0        : (n_mag, 3) equilibrium unit vectors
    frames        : (n_mag, 2, 3) from `tangent_frames`
    target_norms  : (n_mag,) moment magnitudes |M_i|, in mu_B
    mu_eV         : (n_mag,) mu_B * target_norms, in eV/T
    t             : time passed through to `Beff` unchanged; see module
                    docstring -- must be an instant at which the energy is
                    static

    Returns
    -------
    (2*n_mag,) ndarray
    """
    m_hat, r = retract(x, m_hat0, frames)
    M = m_hat * target_norms[:, None]
    n_mag = m_hat0.shape[0]
    g = np.empty(2 * n_mag)
    for i in range(n_mag):
        B_i = np.asarray(model.Beff(t, i, M), float)
        G_i = -mu_eV[i] * B_i
        PG = G_i - m_hat[i] * (m_hat[i] @ G_i)   # P_i(x) @ G_i
        g[2 * i] = frames[i, 0] @ PG / r[i]
        g[2 * i + 1] = frames[i, 1] @ PG / r[i]
    return g


def _energy_tilde(model, x, m_hat0, frames, target_norms, t):
    m_hat, _ = retract(x, m_hat0, frames)
    M = m_hat * target_norms[:, None]
    return float(model.energy(t, M)["total"])


# ---------------------------------------------------------------------------
# Hessian routes
# ---------------------------------------------------------------------------
def _hessian_grad(model, m_hat0, frames, target_norms, mu_eV, t, h):
    """Central differences of `grad_tangent`. 2*(2*n_mag) evaluations."""
    nq = 2 * m_hat0.shape[0]
    H = np.empty((nq, nq))
    for b in range(nq):
        xp = np.zeros(nq)
        xp[b] = h
        xm = np.zeros(nq)
        xm[b] = -h
        gp = grad_tangent(model, xp, m_hat0, frames, target_norms, mu_eV, t)
        gm = grad_tangent(model, xm, m_hat0, frames, target_norms, mu_eV, t)
        H[:, b] = (gp - gm) / (2 * h)
    return 0.5 * (H + H.T)


def _hessian_energy(model, m_hat0, frames, target_norms, t, h):
    """Central second differences of `E_tilde`.

    `2*Nq` diagonal plus `4*Nq*(Nq-1)/2` off-diagonal energy evaluations
    (plus one call for the equilibrium energy used by the diagonal formula).
    """
    nq = 2 * m_hat0.shape[0]
    x0 = np.zeros(nq)
    E0 = _energy_tilde(model, x0, m_hat0, frames, target_norms, t)
    H = np.zeros((nq, nq))

    Eplus = np.empty(nq)
    Eminus = np.empty(nq)
    for a in range(nq):
        ea = np.zeros(nq)
        ea[a] = h
        Eplus[a] = _energy_tilde(model, x0 + ea, m_hat0, frames, target_norms, t)
        Eminus[a] = _energy_tilde(model, x0 - ea, m_hat0, frames, target_norms, t)
        H[a, a] = (Eplus[a] - 2.0 * E0 + Eminus[a]) / h ** 2

    for a in range(nq):
        ea = np.zeros(nq)
        ea[a] = h
        for b in range(a + 1, nq):
            eb = np.zeros(nq)
            eb[b] = h
            Epp = _energy_tilde(model, x0 + ea + eb, m_hat0, frames, target_norms, t)
            Epm = _energy_tilde(model, x0 + ea - eb, m_hat0, frames, target_norms, t)
            Emp = _energy_tilde(model, x0 - ea + eb, m_hat0, frames, target_norms, t)
            Emm = _energy_tilde(model, x0 - ea - eb, m_hat0, frames, target_norms, t)
            val = (Epp - Epm - Emp + Emm) / (4.0 * h ** 2)
            H[a, b] = H[b, a] = val

    return 0.5 * (H + H.T)


def _vectorize_over_trailing_axes(scalar_out, f):
    """Wrap a function of a single flat point so it also accepts a
    `(nq, ...)` batch of points, as required by `scipy.differentiate`."""

    def wrapped(x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            return f(x)
        nq = x.shape[0]
        batch_shape = x.shape[1:]
        flat = x.reshape(nq, -1)
        k = flat.shape[1]
        if scalar_out:
            out = np.empty(k)
            for j in range(k):
                out[j] = f(flat[:, j])
            return out.reshape(batch_shape)
        else:
            out = np.empty((nq, k))
            for j in range(k):
                out[:, j] = f(flat[:, j])
            return out.reshape((nq,) + batch_shape)

    return wrapped


def _hessian_grad_scipy(model, m_hat0, frames, target_norms, mu_eV, t, h):
    # `h` is accepted for interface symmetry with the hand-rolled routes but
    # not passed to scipy: scipy.differentiate's own adaptive step selection
    # (default initial_step=0.5, shrunk by Richardson extrapolation) is more
    # robust here than seeding it with a step already sized for a plain
    # central difference -- seeding a scalar-hessian search that small starves
    # its Richardson extrapolation of a usable step sequence (verified
    # empirically: it converges to a wrong answer for `_hessian_energy_scipy`
    # once `initial_step` drops to ~1e-4).
    from scipy.differentiate import jacobian as sp_jacobian

    nq = 2 * m_hat0.shape[0]
    f = _vectorize_over_trailing_axes(
        False,
        lambda x: grad_tangent(model, x, m_hat0, frames, target_norms, mu_eV, t),
    )
    res = sp_jacobian(f, np.zeros(nq))
    H = np.asarray(res.df, float)
    return 0.5 * (H + H.T)


def _hessian_energy_scipy(model, m_hat0, frames, target_norms, t, h):
    from scipy.differentiate import hessian as sp_hessian

    nq = 2 * m_hat0.shape[0]
    f = _vectorize_over_trailing_axes(
        True,
        lambda x: _energy_tilde(model, x, m_hat0, frames, target_norms, t),
    )
    res = sp_hessian(f, np.zeros(nq))
    H = np.asarray(res.ddf, float)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def calculate_modes(model, M, t=0.0, alpha=0.0, gamma=None, method="grad", h=1e-4,
          g_tol=1e-6, strict=False, soft_tol=1e-6):
    """Uniform (q = 0) magnon modes of `model` at the configuration `M`.

    `M` must be (numerically) an equilibrium of the static energy at the
    given `t`: see the module docstring for what "static" requires of `t`.
    The gradient is checked against the Hessian scale on entry (see
    `g_tol`/`strict` below); it is not itself used to relax the
    configuration. Relax with your own damped LLG integrator first, then
    call this function to check the result and get the spectrum.

    Parameters
    ----------
    model : object exposing `energy(t, M)["total"]` (eV) and
        `Beff(t, n, M)` (Tesla), and optionally `mu_B`, `gamma_e` class
        attributes (see module docstring for fallbacks).
    M : (n_mag, 3) ndarray
        Equilibrium Cartesian moments, magnitude in mu_B (not unit
        vectors). Moment magnitudes are taken from `|M_i|` as supplied here.
    t : float
        Passed through to `energy`/`Beff` unchanged. Caller's
        responsibility that the energy is static at this `t`.
    alpha : float
        Gilbert damping, uniform across sites. 0 for the undamped spectrum.
    gamma : float or None
        Gyromagnetic ratio, rad/(s T). Defaults to `model.gamma_e` (or the
        module fallback) if not given.
    method : {"grad", "grad-scipy", "energy", "energy-scipy"}
        Hessian route. "grad" (default) central-differences the exact
        tangent gradient and is preferred: its error scales as
        `eps_g/h + h^2`, against `eps_E/h^2 + h^2` for the energy route,
        roughly four orders of magnitude better at the same step. The
        "-scipy" routes need SciPy >= 1.15 (`scipy.differentiate`); if the
        import fails, this falls back to the corresponding hand-rolled
        route with a warning, and the backend actually used is recorded in
        the result's `backend` entry.
    h : float
        Finite-difference step, default 1e-4. This assumes a model energy
        accurate to near machine precision; any internal tolerance (Ewald
        sums, iterative solvers, ...) in the model raises the noise floor
        and moves the optimal step up. Run an h-scan on your own model (see
        the module's acceptance test 5) and read the minimum off it.
    g_tol : float
        The linearisation drops `dW/dx` terms only because they multiply
        `grad E(x=0) = 0`. Warn (or raise, if `strict=True`) when
        `max|grad| > g_tol * max|H| * h`. Note a relaxation that stopped on
        `|dM|` can sit at a gradient that is small but not small enough --
        always check this rather than trusting the stopping criterion of
        whatever integrator produced `M`.
    strict : bool
        Raise `RuntimeError` instead of warning when the gradient check
        above fails.
    soft_tol : float
        Relative tolerance, applied against `abs(h_eigvals[-1])`, used for
        three things: the stability flag, the count `n_soft` of `H`
        eigenvalues treated as zero, and (against the largest `|Im|` of the
        full `W @ H` spectrum) flagging a kept eigenvalue's real part as a
        genuine growth rate rather than roundoff. Default 1e-6. A tolerance
        as tight as 1e-8 (appropriate for an exact analytic Hessian) is too
        strict for a finite-differenced `H`, where noise at the `h^2`/`eps/h`
        level can push a genuine zero mode slightly negative and trip a
        false instability alarm at `h` as coarse as 1e-3; loosen or tighten
        alongside `h`.

        For two macrospins at zero field, `n_soft` is a diagnostic in its
        own right: expect 1 for a canted state with easy-plane anisotropy,
        2 for a collinear state with isotropic exchange (or a canted state
        whose exchange derivative vanishes at the equilibrium), and 3 for a
        non-collinear state with no anisotropy at all. An odd `n_soft` with
        no negative `h_eigvals` is a genuinely gapless (Goldstone) mode --
        recheck your expected symmetry count. A negative `h_eigvals[0]`
        beyond `soft_tol` is instead a saddle point of `H` -- re-relax from
        a different starting configuration, or use continuation.

    Returns
    -------
    dict with keys
        omega_rad_s, omega_THz, gamma_rad_s : (n_mag,) ascending in omega
        growth_rate : (n_mag,) == `-Re(eigenvalue)` of the kept branch of
                   `W @ H`, same array as `gamma_rad_s` -- named separately
                   because a nonzero value here at `alpha = 0` is not
                   Gilbert damping but a saddle-point instability, and is
                   otherwise silently indistinguishable from a genuine
                   `omega = 0` zero mode.
        evec_x   : (n_mag, 2*n_mag) complex, tangent coordinates
        evec_m   : (n_mag, n_mag, 3) complex, Cartesian moment deviations;
                   physical motion is `Re[evec_m[k] * exp(-i*omega[k]*t)]`
        frames   : (n_mag, 2, 3), the (e_i1, e_i2) tangent frames
        H, W, grad, h_eigvals, eigvals : intermediate quantities
        n_soft   : int, count of `H` eigenvalues within `soft_tol` of zero
        stable   : bool, `h_eigvals[0] > -soft_tol * |h_eigvals[-1]|`
        backend, mu_eV, method, h

    See the module docstring for the Goldstone-mode, saddle-point and
    complex-eigenvector pitfalls.
    """
    M = np.asarray(M, float)
    n_mag = M.shape[0]
    mu_B, gamma_e = _get_constants(model)
    if gamma is None:
        gamma = gamma_e

    target_norms = np.linalg.norm(M, axis=1)
    m_hat0 = M / target_norms[:, None]
    mu_eV = mu_B * target_norms
    frames = tangent_frames(m_hat0)

    nq = 2 * n_mag
    x0 = np.zeros(nq)
    grad0 = grad_tangent(model, x0, m_hat0, frames, target_norms, mu_eV, t)

    backend = method
    if method == "grad":
        H = _hessian_grad(model, m_hat0, frames, target_norms, mu_eV, t, h)
    elif method == "energy":
        H = _hessian_energy(model, m_hat0, frames, target_norms, t, h)
    elif method == "grad-scipy":
        try:
            H = _hessian_grad_scipy(model, m_hat0, frames, target_norms, mu_eV, t, h)
            backend = "scipy.differentiate"
        except Exception as exc:
            warnings.warn(
                f"grad-scipy backend unavailable ({exc!r}); falling back to "
                "hand-rolled central differences (method='grad')"
            )
            H = _hessian_grad(model, m_hat0, frames, target_norms, mu_eV, t, h)
            backend = "grad (fallback from grad-scipy)"
    elif method == "energy-scipy":
        try:
            H = _hessian_energy_scipy(model, m_hat0, frames, target_norms, t, h)
            backend = "scipy.differentiate"
        except Exception as exc:
            warnings.warn(
                f"energy-scipy backend unavailable ({exc!r}); falling back to "
                "hand-rolled central differences (method='energy')"
            )
            H = _hessian_energy(model, m_hat0, frames, target_norms, t, h)
            backend = "energy (fallback from energy-scipy)"
    else:
        raise ValueError(f"unknown method {method!r}")

    # equilibrium check: the linearisation drops dW/dx only because it
    # multiplies grad E(x0) = 0
    h_scale = float(np.max(np.abs(H))) if H.size else 0.0
    g_max = float(np.max(np.abs(grad0))) if grad0.size else 0.0
    if g_max > g_tol * h_scale * h:
        msg = (
            f"tangent gradient at the supplied configuration is too large "
            f"for the linearisation to be valid: max|grad| = {g_max:.3e} eV "
            f"> g_tol * max|H| * h = {g_tol * h_scale * h:.3e} eV. A "
            "relaxation stopped on |dM| can sit at a gradient that is "
            "small but not small enough; relax further and recheck."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg)

    # block-diagonal symplectic prefactor; orthonormal frame -> no 1/sin(theta)
    W = np.zeros((nq, nq))
    for i in range(n_mag):
        pre = gamma / (mu_eV[i] * (1.0 + alpha ** 2))
        W[2 * i, 2 * i] = -pre * alpha
        W[2 * i, 2 * i + 1] = -pre
        W[2 * i + 1, 2 * i] = pre
        W[2 * i + 1, 2 * i + 1] = -pre * alpha

    # h_eigvals drives the stability flag, n_soft and the Goldstone check
    # below; matrix_rank(W @ H) is unusable for the latter because W's
    # gamma/mu_eV prefactor (~1e15) swamps the rank-revealing threshold that
    # matrix_rank picks from D's own singular values, so it never sees the
    # deficiency. Counting near-zero eigenvalues of H itself sidesteps that
    # scale issue.
    h_eigvals = np.linalg.eigvalsh(H)
    tol_scale = soft_tol * abs(h_eigvals[-1])
    n_soft = int(np.sum(np.abs(h_eigvals) < tol_scale))
    stable = bool(h_eigvals[0] > -tol_scale)

    if not stable:
        warnings.warn(
            f"H has a negative eigenvalue {h_eigvals[0]:.3e} (relative "
            f"{h_eigvals[0] / abs(h_eigvals[-1]):.2e}) beyond soft_tol="
            f"{soft_tol:.1e} -- this configuration is a saddle point of H, "
            "not a minimum. Re-relax from a different starting "
            "configuration, or use numerical continuation from a "
            "known-stable point."
        )
    elif n_soft % 2 == 1:
        warnings.warn(
            f"H has {n_soft} near-zero eigenvalue(s) (relative tolerance "
            f"{soft_tol:.1e}) with none of them negative -- an odd count of "
            "genuinely gapless modes. This is the signature of a defective "
            "(Jordan-block) zero pair of W @ H from a continuous symmetry: "
            "check that this matches your expected symmetry count (see the "
            "calculate_modes() docstring for the standard two-macrospin, zero-field "
            "cases). Its motion is a linear drift in time, not a periodic "
            "orbit -- any trajectory or ellipse built from it is "
            "meaningless."
        )

    D = W @ H
    eigvals, eigvecs = np.linalg.eig(D)

    # eigenvalues come in conjugate pairs -i*omega - Gamma; keep the n_mag
    # with the most negative imaginary part
    order = np.argsort(eigvals.imag)
    keep = order[:n_mag]
    vals_k = eigvals[keep]
    vecs_k = eigvecs[:, keep]

    omega = -vals_k.imag
    decay = -vals_k.real
    idx = np.argsort(omega)
    omega = omega[idx]
    decay = decay[idx]
    vecs_k = vecs_k[:, idx]

    # A saddle point of H gives W @ H a purely real eigenvalue pair --
    # exponential growth, not oscillation -- which prints as omega = 0
    # unless flagged here explicitly.
    imag_scale = float(np.max(np.abs(eigvals.imag))) if eigvals.size else 0.0
    growth_rate = decay
    unstable_modes = np.abs(growth_rate) > soft_tol * imag_scale
    if np.any(unstable_modes):
        warnings.warn(
            f"{int(np.sum(unstable_modes))} of the {n_mag} kept W @ H "
            "eigenvalue(s) have a non-negligible real part -- exponential "
            "growth, not oscillation, printing as omega = 0 unless you "
            f"check growth_rate too. Growth rate(s) = "
            f"{growth_rate[unstable_modes]} rad/s, relative to the largest "
            f"|Im(eigenvalue)| = {imag_scale:.3e} rad/s."
        )

    evec_x = vecs_k.T   # (n_mag, 2*n_mag)
    evec_m = np.zeros((n_mag, n_mag, 3), dtype=complex)
    for k in range(n_mag):
        for i in range(n_mag):
            evec_m[k, i] = (evec_x[k, 2 * i] * frames[i, 0]
                            + evec_x[k, 2 * i + 1] * frames[i, 1])

    return dict(
        omega_rad_s=omega,
        omega_THz=omega / (2.0 * np.pi * 1e12),
        gamma_rad_s=decay,
        growth_rate=growth_rate,
        evec_x=evec_x,
        evec_m=evec_m,
        frames=frames,
        H=H, W=W, grad=grad0,
        h_eigvals=h_eigvals, eigvals=eigvals,
        n_soft=n_soft,
        stable=stable,
        backend=backend, mu_eV=mu_eV, method=method, h=h,
    )


# ---------------------------------------------------------------------------
# Model mixin
# ---------------------------------------------------------------------------
class MagnonModesMixin:
    """
    Mixin class adding q = 0 magnon-mode calculation to the Model, alongside
    MinimizerMixin in minimizers.py. Expects 'self' to have:
      - energy(t, M) -> dict, with a 'total' key, eV
      - Beff(t, n, M) -> ndarray, shape (3,), Tesla
      - mu_B, gamma_e (float) [optional; falls back to the module constants
        MU_B_EV_PER_T, GAMMA_RAD_S_T if absent]
    """

    def calculate_modes(self, M, t=0.0, alpha=0.0, gamma=None, method="grad",
                         h=1e-4, g_tol=1e-6, strict=False, soft_tol=1e-6):
        """Uniform (q = 0) magnon modes of this model at configuration `M`.

        Thin wrapper around the module-level `calculate_modes(model, M,
        ...)` with `self` as the model; see that function's docstring (and
        the module docstring) for the full parameter and return-value
        reference, the static-energy caveat on `t`, and the Goldstone/
        saddle-point pitfalls.
        """
        return calculate_modes(self, M, t=t, alpha=alpha, gamma=gamma,
                                method=method, h=h, g_tol=g_tol,
                                strict=strict, soft_tol=soft_tol)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    class StubModel:
        """Minimal stand-in exposing only `energy(t, M)['total']` and
        `Beff(t, n, M)`, so the tests below never import `model.py`.

            E = f(s) + K_u * sum_i (m_i . z)^2 - sum_i mu_eV_i * B . m_i

        with f(s) = 3 * J1 * s, s = m_hat_0 . m_hat_1, defined only for
        n_mag == 2 (J1 is ignored for n_mag == 1).
        """

        mu_B = MU_B_EV_PER_T
        gamma_e = GAMMA_RAD_S_T

        def __init__(self, J1=0.0, K_u=0.0, B=(0.0, 0.0, 0.0)):
            self.J1 = float(J1)
            self.K_u = float(K_u)
            self.B = np.asarray(B, float)

        def _unpack(self, M):
            M = np.asarray(M, float)
            norms = np.linalg.norm(M, axis=1)
            m_hat = M / norms[:, None]
            mu_eV = MU_B_EV_PER_T * norms
            return m_hat, mu_eV

        def energy(self, t, M):
            m_hat, mu_eV = self._unpack(M)
            n_mag = m_hat.shape[0]
            E = 0.0
            if n_mag == 2 and self.J1 != 0.0:
                s = float(m_hat[0] @ m_hat[1])
                E += 3.0 * self.J1 * s
            E += self.K_u * float(np.sum(m_hat[:, 2] ** 2))
            E -= float(np.sum(mu_eV * (m_hat @ self.B)))
            return {"total": E}

        def Beff(self, t, n, M):
            m_hat, mu_eV = self._unpack(M)
            n_mag = m_hat.shape[0]
            G = np.zeros(3)
            if n_mag == 2 and self.J1 != 0.0:
                other = 1 - n
                G += 3.0 * self.J1 * m_hat[other]
            G += 2.0 * self.K_u * m_hat[n, 2] * np.array([0.0, 0.0, 1.0])
            G -= mu_eV[n] * self.B
            return -G / mu_eV[n]

    n_pass = 0
    n_fail = 0

    def check(name, cond, detail=""):
        global n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"PASS  {name}")
        else:
            n_fail += 1
            print(f"FAIL  {name}  {detail}")

    # -- test 1: single spin at the pole -----------------------------------
    Bz = 2.5
    model1 = StubModel(B=(0.0, 0.0, Bz))
    M1 = np.array([[0.0, 0.0, 1.7]])   # arbitrary moment magnitude
    res1 = calculate_modes(model1, M1, alpha=0.0)
    omega_exact = GAMMA_RAD_S_T * Bz
    rel1 = abs(res1["omega_rad_s"][0] - omega_exact) / omega_exact
    check("1. single spin at the pole", rel1 < 1e-6, f"rel err = {rel1:.3e}")

    # -- test 2: two-spin collinear FM along z, field along z -------------
    J1 = -0.008
    K_u = -0.0005
    mu = 3.0
    B = 10.0
    model2 = StubModel(J1=J1, K_u=K_u, B=(0.0, 0.0, B))
    M2 = np.array([[0.0, 0.0, mu], [0.0, 0.0, mu]])
    res2 = calculate_modes(model2, M2, alpha=0.0)
    mu_eV2 = MU_B_EV_PER_T * mu
    fprime1 = 3.0 * J1   # f(s) = 3*J1*s -> f'(1) = 3*J1
    k_uniform = -2.0 * K_u + mu_eV2 * B
    k_staggered = -2.0 * K_u + mu_eV2 * B - 2.0 * fprime1
    omega_uniform = (GAMMA_RAD_S_T / mu_eV2) * k_uniform
    omega_staggered = (GAMMA_RAD_S_T / mu_eV2) * k_staggered
    expected_THz = np.sort([omega_uniform, omega_staggered]) / (2 * np.pi * 1e12)
    rel2 = np.max(np.abs(res2["omega_THz"] - expected_THz) / expected_THz)
    check("2. two-spin collinear FM, field || z", rel2 < 1e-5,
          f"got {res2['omega_THz']} THz, expected {expected_THz} THz, "
          f"rel err = {rel2:.3e}")
    check("2b. expected numbers match spec (0.441636, 8.188174 THz)",
          np.allclose(expected_THz, [0.441636, 8.188174], rtol=1e-5),
          f"expected_THz = {expected_THz}")

    # -- test 3: unequal moments, permutation invariance --------------------
    mu_pair = (3.0, 2.0)
    M3a = np.array([[0.0, 0.0, mu_pair[0]], [0.0, 0.0, mu_pair[1]]])
    M3b = np.array([[0.0, 0.0, mu_pair[1]], [0.0, 0.0, mu_pair[0]]])
    res3a = calculate_modes(model2, M3a, alpha=0.0)
    res3b = calculate_modes(model2, M3b, alpha=0.0)
    changed = not np.allclose(res3a["omega_THz"], res2["omega_THz"], rtol=1e-3)
    permuted_ok = np.allclose(res3a["omega_THz"], res3b["omega_THz"], rtol=1e-8)
    check("3. unequal moments change the spectrum", changed)
    check("3b. unequal moments: permutation invariant", permuted_ok)

    # -- test 4: method agreement on the stiff mode of test 2 --------------
    stiff_ref = res2["omega_THz"][-1]
    agree = True
    got = {}
    for meth in ("grad", "energy", "grad-scipy", "energy-scipy"):
        try:
            r = calculate_modes(model2, M2, alpha=0.0, method=meth)
        except Exception as exc:
            print(f"  (skipping method={meth}: {exc!r})")
            continue
        got[meth] = r["omega_THz"][-1]
        rel = abs(r["omega_THz"][-1] - stiff_ref) / stiff_ref
        agree &= rel < 1e-5
    check("4. method agreement on stiff mode", agree, f"{got}")

    # -- test 5: step-size convergence, method='grad' ------------------------
    # Compare against the exact analytic reference (not stiff_ref, which was
    # itself computed at h=1e-4 -- comparing against it would make the
    # h=1e-4 point a trivial self-comparison).
    hs = [1e-2, 1e-3, 1e-4, 1e-5]
    errs = []
    for hh in hs:
        r = calculate_modes(model2, M2, alpha=0.0, method="grad", h=hh)
        errs.append(abs(r["omega_THz"][-1] - expected_THz[-1]))
    ratios = [errs[i] / errs[i + 1] if errs[i + 1] > 0 else float("inf")
              for i in range(len(errs) - 1)]
    clean_h2 = sum(30.0 < ratio < 300.0 for ratio in ratios) >= 2
    check("5. h^2 convergence over >= 2 decades", clean_h2,
          f"errs = {errs}, ratios = {ratios} (expect ~100)")

    # -- test 6: cross-check against magnon_modes.py, if importable --------
    try:
        import magnon_modes as _mm

        K_u_plane = 0.0010
        J_coeffs = {1: 0.008, 2: 0.005}
        mu2 = np.array([3.0, 3.0])
        twospin = _mm.TwoSpinModel(J_coeffs, K_u=K_u_plane, mu=mu2, B=(0, 0, 0))
        from scipy.optimize import brentq
        s0 = brentq(twospin.df, -0.999, 0.999)
        psi0 = np.arccos(s0)
        ang0 = np.array([np.pi / 2, +psi0 / 2, np.pi / 2, -psi0 / 2])
        ang_eq, _ = twospin.find_equilibrium(ang0)
        ref = twospin.modes(ang_eq, method="analytic")

        class _WrapModel:
            """energy_cart is exact and unconstrained (not internally
            renormalised), so its own analytic Cartesian gradient
            `_cart_grad_hess` gives an exact Beff here -- no finite
            differences needed, keeping this cross-check to 1e-6."""

            mu_B = MU_B_EV_PER_T
            gamma_e = GAMMA_RAD_S_T

            def __init__(self, inner):
                self.inner = inner

            def energy(self, t, M):
                M = np.asarray(M, float)
                mhat = M / np.linalg.norm(M, axis=1, keepdims=True)
                return {"total": self.inner.energy_cart(mhat)}

            def Beff(self, t, n, M):
                M = np.asarray(M, float)
                norms = np.linalg.norm(M, axis=1)
                mhat = M / norms[:, None]
                G, _ = self.inner._cart_grad_hess(mhat)
                mu_eV_n = MU_B_EV_PER_T * norms[n]
                return -G[n] / mu_eV_n

        wrapped = _WrapModel(twospin)
        M_eq = _mm.angles_to_M(ang_eq, mu2)
        res6 = calculate_modes(wrapped, M_eq, alpha=0.0)
        rel6 = abs(res6["omega_THz"][-1] - ref["omega_THz"][-1]) / ref["omega_THz"][-1]
        check("6. cross-check vs magnon_modes.TwoSpinModel (stiff mode)",
              rel6 < 1e-6, f"rel err = {rel6:.3e}")
        soft_num = np.min(res6["h_eigvals"])
        print(f"  (soft mode: {res6['omega_THz'][0]:.6e} THz vs analytic "
              f"{ref['omega_THz'][0]:.6e} THz -- expected to differ at the "
              f"1e-4 THz level, finite-difference noise on a Goldstone "
              f"mode; smallest H eigenvalue = {soft_num:.3e})")
    except ImportError:
        print("SKIP  6. magnon_modes.py not importable, skipping cross-check")

    # -- tests 7-8: marginal easy-plane AFM -- n_soft as a diagnostic ------
    # B = 0 always gives one guaranteed Goldstone mode (global rotation about
    # z is a symmetry of exchange + easy-plane anisotropy regardless of
    # configuration). At the exactly collinear-antiparallel equilibrium
    # (both spins in-plane, s = m_0.m_1 = -1), a *second* zero mode appears
    # iff f'(-1) = 0: J_coeffs = {1: 0.0005, 2: 0.0001} solves
    # f'(-1) = 3*J1 - 15*J2 = 0 exactly, so n_soft should come out at 2 with
    # both frequencies at zero. Nudging J1 up by 0.00051 - 0.0005 lifts that
    # second mode to a small finite frequency, leaving n_soft = 1 (just the
    # unavoidable global-rotation Goldstone mode).
    class LegendreStubModel:
        """f(s) = sum_n J_n (2n+1) P_n(s), s = m_hat_0 . m_hat_1 (n_mag == 2
        only); E = f(s) + K_u * sum_i (m_i.z)^2 - sum_i mu_eV_i * B.m_i."""

        mu_B = MU_B_EV_PER_T
        gamma_e = GAMMA_RAD_S_T

        def __init__(self, J_coeffs, K_u=0.0, B=(0.0, 0.0, 0.0)):
            from numpy.polynomial import legendre as npleg
            nmax = max(J_coeffs) if J_coeffs else 0
            c = np.zeros(nmax + 1)
            for n, J in J_coeffs.items():
                c[n] = J * (2 * n + 1)
            self._npleg = npleg
            self._c = c
            self._c1 = npleg.legder(c, 1) if nmax >= 1 else np.zeros(1)
            self.K_u = float(K_u)
            self.B = np.asarray(B, float)

        def f(self, s):
            return float(self._npleg.legval(s, self._c))

        def df(self, s):
            return float(self._npleg.legval(s, self._c1))

        def _unpack(self, M):
            M = np.asarray(M, float)
            norms = np.linalg.norm(M, axis=1)
            m_hat = M / norms[:, None]
            mu_eV = MU_B_EV_PER_T * norms
            return m_hat, mu_eV

        def energy(self, t, M):
            m_hat, mu_eV = self._unpack(M)
            s = float(m_hat[0] @ m_hat[1])
            E = self.f(s)
            E += self.K_u * float(np.sum(m_hat[:, 2] ** 2))
            E -= float(np.sum(mu_eV * (m_hat @ self.B)))
            return {"total": E}

        def Beff(self, t, n, M):
            m_hat, mu_eV = self._unpack(M)
            s = float(m_hat[0] @ m_hat[1])
            other = 1 - n
            G = self.df(s) * m_hat[other]
            G += 2.0 * self.K_u * m_hat[n, 2] * np.array([0.0, 0.0, 1.0])
            G -= mu_eV[n] * self.B
            return -G / mu_eV[n]

    K_u_ep = 0.0010
    mu_ep = 3.0
    M_afm = np.array([[mu_ep, 0.0, 0.0], [-mu_ep, 0.0, 0.0]])  # s = -1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model_marginal = LegendreStubModel({1: 0.0005, 2: 0.0001}, K_u=K_u_ep)
        res_marginal = calculate_modes(model_marginal, M_afm, alpha=0.0)
    saddle_warned = any("saddle" in str(w.message) for w in caught)
    check("7. marginal easy-plane AFM: n_soft == 2",
          res_marginal["n_soft"] == 2, f"n_soft = {res_marginal['n_soft']}")
    # Both this eigendirection's H eigenvalue and W @ H's real/imaginary
    # split are exactly at a defective double root here (f'(-1) = 0 to
    # machine precision), so the central-difference Hessian sees a purely
    # quartic landscape along it: the O(h^2) discretisation floor this
    # produces is a real, deterministic artifact (not roundoff noise), and
    # scales down cleanly with h -- but its size, and how eig() happens to
    # split it between the real and imaginary parts, is still well above
    # 1e-6 THz at the default h=1e-4. 3e-4 THz stays far below the 0.056
    # THz signal of the detuned case (test 8) while comfortably clearing
    # that floor.
    check("7b. marginal easy-plane AFM: both frequencies ~ 0",
          np.allclose(res_marginal["omega_THz"], 0.0, atol=3e-4),
          f"omega_THz = {res_marginal['omega_THz']}")
    check("7c. marginal easy-plane AFM: no saddle warning",
          not saddle_warned and res_marginal["stable"],
          f"stable = {res_marginal['stable']}, warnings = "
          f"{[str(w.message) for w in caught]}")

    model_detuned = LegendreStubModel({1: 0.00051, 2: 0.0001}, K_u=K_u_ep)
    res_detuned = calculate_modes(model_detuned, M_afm, alpha=0.0)
    check("8. detuned easy-plane AFM: n_soft == 1",
          res_detuned["n_soft"] == 1, f"n_soft = {res_detuned['n_soft']}")
    check("8b. detuned easy-plane AFM: second frequency ~ 0.056 THz",
          abs(res_detuned["omega_THz"][-1] - 0.056) < 0.002,
          f"omega_THz = {res_detuned['omega_THz']}")

    print(f"\n{n_pass} passed, {n_fail} failed")
    import sys
    sys.exit(0 if n_fail == 0 else 1)
