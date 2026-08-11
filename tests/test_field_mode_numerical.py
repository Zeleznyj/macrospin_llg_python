"""
Tests for Model.effective_field_numerical and the field_mode dispatch
(analytic Beff vs. central-difference field on the total energy).

See effective_field_numerical() / effective_field() / field_mode in model.py.

The radial component of Beff (the part parallel to M_n) never affects the LLG
dynamics: M x Beff discards it. The analytic field expressions are not derived
to make this component vanish, even though the true energy gradient of every
scale-invariant term (exchange, DMI, anisotropy) is exactly transverse. So the
tangential component is checked with a tight tolerance everywhere, while the
radial component is only expected to match for the (non scale-invariant)
static field term; mismatches elsewhere are recorded as xfail.
"""
import numpy as np
import pytest

from macrospin_llg.model import Model

N_MAG = 4


def _transverse(B, M):
    n = M / np.linalg.norm(M, axis=1, keepdims=True)
    return B - n * np.sum(B * n, axis=1, keepdims=True)


def _radial(B, M):
    n = M / np.linalg.norm(M, axis=1, keepdims=True)
    return np.sum(B * n, axis=1)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def random_state(rng):
    """A few random, non-symmetric unit moments."""
    def _make(n_mag=N_MAG):
        M = rng.normal(size=(n_mag, 3))
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        return M
    return _make


def _build_model(term, rng, n_mag=N_MAG):
    m = Model(n_mag)
    if term in ("exchange", "all"):
        for i in range(n_mag):
            for j in range(i + 1, n_mag):
                m.add_exchange(i, j, rng.normal(size=(3, 3)) * 1e-3)
    if term in ("dmi", "all"):
        for i in range(n_mag):
            for j in range(i + 1, n_mag):
                m.add_DMI(i, j, rng.normal(size=3) * 1e-4)
    if term in ("ani2", "all"):
        for i in range(n_mag):
            m.add_ani2(i, rng.normal() * 1e-4, rng.normal(size=3))
    if term in ("ani4", "all"):
        for i in range(n_mag):
            m.add_ani4(i, rng.normal() * 1e-4)
    if term in ("ani6", "all"):
        for i in range(n_mag):
            m.K6[i] = rng.normal() * 1e-4
    if term in ("b_field", "all"):
        for i in range(n_mag):
            m.add_B(i, rng.normal(size=3) * 1e-2)
    return m


def _fields(model, M):
    model.field_mode = "analytic"
    B_analytic = model.effective_field(0.0, M)
    model.field_mode = "numerical"
    B_numerical = model.effective_field(0.0, M)
    model.field_mode = "analytic"
    return B_analytic, B_numerical


@pytest.mark.parametrize("term", ["exchange", "dmi", "ani2", "ani4", "ani6", "b_field", "all"])
def test_tangential_field_matches_analytic(rng, random_state, term):
    m = _build_model(term, rng)
    M = random_state()
    B_analytic, B_numerical = _fields(m, M)
    np.testing.assert_allclose(
        _transverse(B_numerical, M), _transverse(B_analytic, M),
        rtol=1e-7, atol=1e-10,
        err_msg=f"{term}: tangential field mismatch between analytic and numerical",
    )


def test_bfield_radial_matches_analytic(rng, random_state):
    m = _build_model("b_field", rng)
    M = random_state()
    B_analytic, B_numerical = _fields(m, M)
    np.testing.assert_allclose(_radial(B_analytic, M), _radial(B_numerical, M), rtol=1e-6, atol=1e-10)


# These terms are exactly scale-invariant in each M_n (E is homogeneous of
# degree 0), so the true energy gradient is purely transverse. The analytic
# Beff expressions were derived assuming |M_n| = 1 and were not constructed
# to zero out the radial component for general |M_n|, so they carry a
# spurious radial piece that the numerical (energy-exact) field does not.
# Harmless for dynamics since M x Beff discards it regardless -- recorded as
# xfail per-term rather than silently tolerated.
@pytest.mark.parametrize("term", [
    pytest.param("exchange", marks=pytest.mark.xfail(strict=True, reason="B_ex substitutes |M_n|=1; radial component differs")),
    pytest.param("dmi", marks=pytest.mark.xfail(strict=True, reason="B_DMI substitutes |M_n|=1; radial component differs")),
    pytest.param("ani2", marks=pytest.mark.xfail(strict=True, reason="B_ani (K2 term) substitutes |M_n|=1; radial component differs")),
    pytest.param("ani4", marks=pytest.mark.xfail(strict=True, reason="B_ani (K4 term) substitutes |M_n|=1; radial component differs")),
    pytest.param("ani6", marks=pytest.mark.xfail(strict=True, reason="B_ani (K6 term) substitutes |M_n|=1; radial component differs")),
])
def test_radial_field_matches_analytic(rng, random_state, term):
    m = _build_model(term, rng)
    M = random_state()
    B_analytic, B_numerical = _fields(m, M)
    np.testing.assert_allclose(_radial(B_analytic, M), _radial(B_numerical, M), rtol=1e-6, atol=1e-10)


def test_constrained_numerical_field_is_radially_zero(rng, random_state):
    """constrained=True differentiates E(M / |M|), which is manifestly
    independent of |M|, so its gradient is exactly transverse."""
    m = _build_model("all", rng)
    M = random_state()
    B_constrained = m.effective_field_numerical(0.0, M, constrained=True)
    np.testing.assert_allclose(_radial(B_constrained, M), 0.0, atol=1e-6)


def test_step_size_convergence(rng, random_state):
    """Error vs. the analytic (tangential) field should fall roughly as h^2
    from h=1e-3 to h=1e-5, then flatten out as roundoff takes over."""
    m = _build_model("exchange", rng)
    M = random_state()
    B_analytic = m.effective_field(0.0, M)
    t_analytic = _transverse(B_analytic, M)

    errs = {}
    for h in (1e-3, 1e-4, 1e-5):
        B_num = m.effective_field_numerical(0.0, M, h=h)
        errs[h] = np.max(np.abs(_transverse(B_num, M) - t_analytic))

    # Roughly quadratic convergence: each decade in h should reduce the error
    # by roughly two decades. Allow generous margins since this is a rough check.
    ratio_1 = errs[1e-3] / errs[1e-4]
    ratio_2 = errs[1e-4] / errs[1e-5]
    assert 30 < ratio_1 < 300, f"expected ~h^2 convergence, got ratio {ratio_1}"
    assert 30 < ratio_2 < 300, f"expected ~h^2 convergence, got ratio {ratio_2}"


def test_field_mode_validation():
    m = Model(2)
    assert m.field_mode == "analytic"
    m.field_mode = "numerical"
    assert m.field_mode == "numerical"
    with pytest.raises(ValueError):
        m.field_mode = "bogus"
    with pytest.raises(ValueError):
        Model(2, field_mode="bogus")


def test_beff_only_custom_term_raises(rng, random_state):
    m = Model(2)
    m.add_custom_interaction("stray_field", beff_per_atom=[lambda t, M: np.array([0.0, 0.0, 1e-3]), None])
    M = random_state(2)
    with pytest.raises(ValueError, match="stray_field"):
        m.effective_field_numerical(0.0, M)


def test_llg_integration_agrees_between_modes():
    """A short integration run in both modes should agree to a loose
    tolerance -- this catches a broken dispatch, not a broken gradient."""
    m = Model(2)
    m.add_exchange(-1, -1, 0.05)
    m.add_B(0, [0.0, 0.0, 0.05])
    m.add_B(1, [0.0, 0.0, -0.05])
    m.ag = 0.05
    M0 = np.array([[1.0, 0.3, 0.0], [-1.0, -0.3, 0.1]])
    t_eval = np.linspace(0.0, 1e-3, 200)

    m.field_mode = "analytic"
    sol_analytic = m.solve_LLG(1e-3, M0, atol=1e-7, rtol=1e-7, t_eval=t_eval)

    m.field_mode = "numerical"
    sol_numerical = m.solve_LLG(1e-3, M0, atol=1e-7, rtol=1e-7, t_eval=t_eval)

    np.testing.assert_allclose(sol_numerical.M, sol_analytic.M, rtol=1e-3, atol=1e-4)
