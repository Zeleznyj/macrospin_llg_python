"""
Regression test: Beff must equal -(1/mu_B) * dE/dM for every energy term.

Since every term is normalised by |M_n|, E is homogeneous of degree 0 in
each M_n, so the true gradient is purely transverse to M_n. Beff (as coded)
drops the longitudinal projector, so we project both sides onto the plane
perpendicular to M_n before comparing -- that's the component that actually
enters the LLG via M x Beff.

Drop this in as e.g. tests/test_field_consistency.py. Adjust the import
below to match your package layout if needed.
"""
import numpy as np
import pytest

from macrospin_llg.model import Model

MU_B = Model.mu_B
H = 1e-6         # finite-difference step
RTOL = 1e-6      # relative tolerance on the transverse field match


def _num_grad(efun, M, n, h=H):
    """Central-difference dE/dM_n (3,)."""
    g = np.zeros(3)
    for k in range(3):
        Mp, Mm = M.copy(), M.copy()
        Mp[n, k] += h
        Mm[n, k] -= h
        g[k] = (efun(Mp) - efun(Mm)) / (2 * h)
    return g


def _transverse(v, Mn):
    u = Mn / np.linalg.norm(Mn)
    return v - np.dot(v, u) * u


def _assert_beff_matches_energy(model, efun, bfun, M, label):
    for n in range(model.n_mag):
        b_num = _transverse(-_num_grad(efun, M, n) / MU_B, M[n])
        b_cod = _transverse(bfun(n, M), M[n])
        scale = max(np.max(np.abs(b_num)), 1e-30)
        err = np.max(np.abs(b_cod - b_num)) / scale
        assert err < RTOL, (
            f"{label}: Beff disagrees with -dE/dM at site {n} "
            f"(rel err {err:.3e}, code={b_cod}, numeric={b_num})"
        )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def random_state(rng):
    """A few random unit moments."""
    def _make(n_mag):
        M = rng.normal(size=(n_mag, 3))
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        return M
    return _make


def test_exchange_field_matches_energy(rng, random_state):
    n_mag = 3
    m = Model(n_mag)
    for i in range(n_mag):
        for j in range(i + 1, n_mag):
            A = rng.normal(size=(3, 3)) * 1e-3
            A = 0.5 * (A + A.T)  # symmetric exchange tensor
            m.add_exchange(i, j, A)
    M = random_state(n_mag)
    _assert_beff_matches_energy(m, m._energy_exchange, m.B_ex, M, "exchange")


def test_exchange_field_matches_energy_nonsymmetric_J(rng, random_state):
    """Exposes the add_exchange mirror-assignment issue for a
    non-symmetric per-pair exchange tensor (J[:,:,j,i] should be J.T,
    not J)."""
    n_mag = 3
    m = Model(n_mag)
    for i in range(n_mag):
        for j in range(i + 1, n_mag):
            m.add_exchange(i, j, rng.normal(size=(3, 3)) * 1e-3)
    M = random_state(n_mag)
    _assert_beff_matches_energy(m, m._energy_exchange, m.B_ex, M, "exchange (non-symmetric J)")


def test_dmi_field_matches_energy(rng, random_state):
    """Currently expected to FAIL: B_DMI is missing the factor of 2 that
    B_ex has (see B_ex vs B_DMI in model.py). Remove xfail once fixed."""
    n_mag = 3
    m = Model(n_mag)
    for i in range(n_mag):
        for j in range(i + 1, n_mag):
            m.add_DMI(i, j, rng.normal(size=3) * 1e-4)
    M = random_state(n_mag)
    _assert_beff_matches_energy(m, m._energy_dmi, m.B_DMI, M, "DMI")


def test_anisotropy_field_matches_energy(rng, random_state):
    n_mag = 3
    m = Model(n_mag)
    for i in range(n_mag):
        m.add_ani2(i, rng.normal() * 1e-4, rng.normal(size=3))
        m.add_ani4(i, rng.normal() * 1e-4)
        m.K6[i] = rng.normal() * 1e-4
    M = random_state(n_mag)
    _assert_beff_matches_energy(m, m._energy_anisotropy, m.B_ani, M, "anisotropy")


def test_total_beff_matches_total_energy(rng, random_state):
    """Sanity check on the combined Beff() against the total energy(),
    for a model with exchange + DMI + anisotropy + external B all active."""
    n_mag = 3
    m = Model(n_mag)
    for i in range(n_mag):
        for j in range(i + 1, n_mag):
            A = rng.normal(size=(3, 3)) * 1e-3
            m.add_exchange(i, j, 0.5 * (A + A.T))
            m.add_DMI(i, j, rng.normal(size=3) * 1e-4)
        m.add_ani2(i, rng.normal() * 1e-4, rng.normal(size=3))
        m.add_B(i, rng.normal(size=3) * 1e-2)

    M = random_state(n_mag)

    def total_e(MM):
        return m.energy(0.0, MM)['total']

    def total_b(n, MM):
        return m.Beff(0.0, n, MM)

    # NOTE: will fail until the DMI factor-of-2 fix lands, since it's
    # summed into the total.
    _assert_beff_matches_energy(m, total_e, total_b, M, "total")
