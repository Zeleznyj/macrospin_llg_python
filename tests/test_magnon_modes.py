"""
Regression tests for Model.calculate_modes() (the MagnonModesMixin added by
macrospin_llg/magnon_modes_numeric.py), exercised through the real Model
class rather than the stand-in models used by that module's own
`if __name__ == "__main__":` self-test.

That self-test (run directly with `python -m macrospin_llg.magnon_modes_numeric`,
or `python macrospin_llg/magnon_modes_numeric.py`) already validates the
tangent-frame/retraction math itself against hand-written stub models and
closed-form results; it deliberately avoids importing Model so it has no
dependency on this package's own field conventions. The tests below instead
check that Model's actual `energy`/`Beff` wiring, reached through
`add_exchange`/`add_ani2`/`add_B`, reproduces the same known-good results
via the real `Model.calculate_modes()` entry point -- i.e. they cover the
integration, not the underlying math.

Unit conversions from the stub-model conventions used in
magnon_modes_numeric.py's self-test to Model's Hamiltonian conventions
(see model.py's `_energy_exchange`/`_energy_anisotropy`/`_energy_b_field`):

  - f(s) = 3*J1*s (linear exchange, s = m_hat_0 . m_hat_1)
      <-> Model.add_exchange(0, 1, 1.5*J1)
    since for n_mag=2, Model's E_exchange = sum_{i!=j} Mi @ (J*I) @ Mj /
    (|Mi| |Mj|) = 2*J*s (both ordered pairs (0,1) and (1,0) contribute),
    so matching 3*J1*s requires 2*J = 3*J1, i.e. J = 1.5*J1.
  - E_ani = K_u * (m_hat . z)^2 per site
      <-> Model.add_ani2(-1, -2*K_u, (0, 0, 1))
    since Model's E_ani = -K2/2 * (m_hat . A2)^2 per site, so K_u = -K2/2.
  - The Zeeman term matches directly: Model.add_B(-1, B) is
    -mu_B*|M_i|*(m_hat_i . B) summed, the same as the stub's
    -mu_eV_i * (m_hat_i . B).
"""

import numpy as np
import pytest

from macrospin_llg.model import Model

J1_TEST = -0.008
K_U_TEST = -0.0005
MU_TEST = 3.0
B_TEST = 10.0
EXPECTED_THZ = np.array([0.441636, 8.188174])  # uniform, staggered


def _two_spin_fm(J1, K_u, mu=(MU_TEST, MU_TEST), B=(0.0, 0.0, B_TEST)):
    """Model(2) matching magnon_modes_numeric.py's self-test StubModel:
    E = f(s) + K_u * sum_i (m_i.z)^2 - sum_i mu_eV_i * B.m_i, f(s) = 3*J1*s.
    """
    m = Model(2)
    m.add_exchange(0, 1, 1.5 * J1)
    m.add_ani2(-1, -2.0 * K_u, (0.0, 0.0, 1.0))
    m.add_B(-1, B)
    M = np.array([[0.0, 0.0, mu[0]], [0.0, 0.0, mu[1]]])
    return m, M


def test_single_spin_at_pole():
    """n_mag = 1, E = -mu_eV*B.m_hat, m_hat = z_hat: exact omega = gamma*Bz,
    independent of the moment magnitude -- the case spherical-angle code
    cannot handle at all."""
    Bz = 2.5
    m = Model(1)
    m.add_B(0, (0.0, 0.0, Bz))
    M = np.array([[0.0, 0.0, 1.7]])  # arbitrary moment magnitude

    res = m.calculate_modes(M, alpha=0.0)

    omega_exact = Model.gamma_e * Bz
    rel = abs(res["omega_rad_s"][0] - omega_exact) / omega_exact
    assert rel < 1e-6


def test_two_spin_collinear_fm_matches_closed_form():
    """Both moments along +z, field along z: uniform/staggered frequencies
    have a known closed form (k_uniform = -2*K_u + mu_eV*B, k_staggered =
    k_uniform - 2*f'(1)), evaluating to 0.441636 and 8.188174 THz for the
    parameters here."""
    m, M = _two_spin_fm(J1_TEST, K_U_TEST)

    res = m.calculate_modes(M, alpha=0.0)

    rel = np.max(np.abs(res["omega_THz"] - EXPECTED_THZ) / EXPECTED_THZ)
    assert rel < 1e-5, f"got {res['omega_THz']} THz, expected {EXPECTED_THZ} THz"


def test_unequal_moments_change_spectrum_and_are_permutation_invariant():
    """Guards against a per-site indexing error in W or mu_eV: unequal
    moments must change the spectrum, and swapping the two sites must not."""
    m_eq, M_eq = _two_spin_fm(J1_TEST, K_U_TEST, mu=(3.0, 3.0))
    res_eq = m_eq.calculate_modes(M_eq, alpha=0.0)

    m_a, M_a = _two_spin_fm(J1_TEST, K_U_TEST, mu=(3.0, 2.0))
    m_b, M_b = _two_spin_fm(J1_TEST, K_U_TEST, mu=(2.0, 3.0))
    res_a = m_a.calculate_modes(M_a, alpha=0.0)
    res_b = m_b.calculate_modes(M_b, alpha=0.0)

    assert not np.allclose(res_a["omega_THz"], res_eq["omega_THz"], rtol=1e-3)
    np.testing.assert_allclose(res_a["omega_THz"], res_b["omega_THz"], rtol=1e-8)


def test_method_agreement_on_stiff_mode():
    """All four `method` values should agree to 1e-5 relative on the stiff
    mode."""
    m, M = _two_spin_fm(J1_TEST, K_U_TEST)
    stiff_ref = m.calculate_modes(M, alpha=0.0)["omega_THz"][-1]

    for method in ("grad", "energy", "grad-scipy", "energy-scipy"):
        res = m.calculate_modes(M, alpha=0.0, method=method)
        rel = abs(res["omega_THz"][-1] - stiff_ref) / stiff_ref
        assert rel < 1e-5, f"method={method}: rel err {rel:.3e}"


def test_h_squared_convergence():
    """h-scan over 1e-2 .. 1e-5 on the default 'grad' route: clean h^2
    convergence over at least two decades before the round-off floor."""
    m, M = _two_spin_fm(J1_TEST, K_U_TEST)

    errs = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5):
        res = m.calculate_modes(M, alpha=0.0, method="grad", h=h)
        errs.append(abs(res["omega_THz"][-1] - EXPECTED_THZ[-1]))

    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1) if errs[i + 1] > 0]
    assert sum(30.0 < r < 300.0 for r in ratios) >= 2, f"errs={errs}, ratios={ratios}"


def test_collinear_afm_goldstone_diagnostics():
    """B = 0, easy-plane anisotropy, collinear-antiparallel equilibrium:
    global rotation about z is always a symmetry here, giving exactly one
    unavoidable Goldstone mode. Checks that the n_soft/stable diagnostics
    added to calculate_modes() see it, and that the finite second (staggered)
    frequency matches its closed form."""
    J1, K_u, mu = 0.008, 0.0010, MU_TEST
    m = Model(2)
    m.add_exchange(0, 1, 1.5 * J1)
    m.add_ani2(-1, -2.0 * K_u, (0.0, 0.0, 1.0))
    M = np.array([[mu, 0.0, 0.0], [-mu, 0.0, 0.0]])  # s = m_0.m_1 = -1

    # A lone Goldstone mode is numerically delicate (see the module's
    # Goldstone-mode pitfall): expect the near-zero-eigenvalue and
    # non-negligible-growth-rate warnings, not an error.
    with pytest.warns(UserWarning):
        res = m.calculate_modes(M, alpha=0.0)

    assert res["n_soft"] == 1
    assert res["stable"]
    assert abs(res["omega_THz"][0]) < 1e-3  # Goldstone mode, ~0 up to noise

    mu_eV = Model.mu_B * mu
    expected_staggered_THz = (
        (2.0 * Model.gamma_e / mu_eV) * np.sqrt(3.0 * J1 * K_u) / (2.0 * np.pi * 1e12)
    )
    rel = abs(res["omega_THz"][-1] - expected_staggered_THz) / expected_staggered_THz
    assert rel < 1e-5


if __name__ == "__main__":
    test_single_spin_at_pole()
    test_two_spin_collinear_fm_matches_closed_form()
    test_unequal_moments_change_spectrum_and_are_permutation_invariant()
    test_method_agreement_on_stiff_mode()
    test_h_squared_convergence()
    test_collinear_afm_goldstone_diagnostics()
    print("All tests passed.")
