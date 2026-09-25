"""
Rate-equation models of charge-carrier recombination for fitting TRPL decays.

Each model is written as three functions:

    <Name>_Model(t, y, args)   right-hand side of the ODE system (Diffrax signature)
    solve_<Name>(t, ...)       solves the ODE system at the times in ``t``
    TRPL_<Name>(t, ..., bkg)   returns (log10 TRPL signal, carrier densities...)

The TRPL signal is normalised to its value at t = 0 and a background ``bkg`` is
added (before normalisation for DT and DTDeepVar, after it for the others). The
carrier densities are returned in the order n, p, trapped electrons.

Models
------
ABC           Free-carrier model: first-order trapping (A), bimolecular (B) and
              Auger (C) recombination. ``TRPL_AB`` is the same model without Auger.
BTD           Bimolecular-Trapping-Detrapping model with Auger, a single trap
              state of density NT, detrapping to the conduction band and
              depopulation to the valence band.
DT            Dual-trap model (DOI: 10.1103/PRXEnergy.4.013001). A shallow trap
              with capture and emission (detrapping only) and a deep trap that
              removes carriers non-radiatively. Neither trap saturates.
DTShallowVar  Dual-trap model where the shallow trap (detrapping active, no
              depopulation) has a finite density, so capture scales with (NT - nT).
DTDeepVar     Dual-trap model where the deep trap (depopulation active, no
              detrapping) has a finite density, so capture scales with (NT - nT).
ShallowTrapVar
              Single shallow trap with finite density and Auger recombination.
FullREM       Full Shockley-Read-Hall rate-equation model with two traps
              (rate equations only, no solver).

Units are cm^-3 for densities and ns for time throughout.

Plotting and analysis helpers used by the simulation notebooks (``colorFader``,
``diff_lifetime``, ``diff_constant``, ``relative_QFLS``) are at the end of the file.
"""

import os

# A failed ODE solve (e.g. NaN in the implicit solver's linear solve for an
# extreme parameter draw) returns NaN instead of raising, so NUTS rejects the
# step rather than the whole run crashing. Must be set before equinox is imported.
os.environ.setdefault("EQX_ON_ERROR", "nan")

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# =============================================================================
# ABC model
# =============================================================================

def ABC_Model(t, n, args):
    """
    Rate equation for the ABC model.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    n : jnp.ndarray
        Free-electron density [n].
    args : jnp.ndarray
        [k_A, k_B, k_C].

    Returns
    -------
    jnp.ndarray
        dn/dt.
    """
    k_A, k_B, k_C = args
    return -k_A * n - k_B * n**2 - k_C * n**3


@jax.jit
def solve_ABC(t, n_0, k_A, k_B, k_C):
    """
    Solve the ABC model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    n_0 : float
        Initial free-electron density (cm^-3).
    k_A : float
        First-order (SRH) rate constant (ns^-1).
    k_B : float
        Bimolecular rate constant (cm^3 ns^-1).
    k_C : float
        Auger rate constant (cm^6 ns^-1).

    Returns
    -------
    diffrax.Solution
        ``sol.ys[:, 0]`` is n(t).
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(ABC_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=0.0002,
        y0=jnp.array([n_0]),
        args=jnp.array([k_A, k_B, k_C]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        max_steps=100000,
        throw=False,
    )


@jax.jit
def TRPL_ABC(t, n_0, k_A, k_B, k_C, bkg):
    """
    TRPL signal for the ABC model, normalised to t = 0.

    Parameters
    ----------
    t, n_0, k_A, k_B, k_C
        See ``solve_ABC``.
    bkg : float
        Background added to the normalised signal.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n).
    """
    sol = solve_ABC(t, n_0, k_A, k_B, k_C)

    n = sol.ys[:, 0]

    sig = k_B * n**2
    sig = sig / sig[0]
    return jnp.log10(sig + bkg), n


@jax.jit
def TRPL_AB(t, n_0, k_A, k_B, bkg):
    """
    TRPL signal for the ABC model without Auger recombination (k_C = 0).

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n).
    """
    return TRPL_ABC(t, n_0, k_A, k_B, 0.0, bkg)


# =============================================================================
# BTD model (Bimolecular-Trapping-Detrapping)
# =============================================================================

def BTD_Model(t, y, args):
    """
    Rate equations for the BTD model.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n_e, n_t, n_h]: free electrons, trapped electrons and
        photo-generated holes.
    args : jnp.ndarray
        [ka, kt, kb, kdt, kdp, NT, p0].

    Returns
    -------
    jnp.ndarray
        [dn_e/dt, dn_t/dt, dn_h/dt].
    """
    ka, kt, kb, kdt, kdp, NT, p0 = args

    # Keep densities within a physical range
    n_e = jnp.clip(y[0], 1e-10, 1e20)
    n_t = jnp.clip(y[1], 1e-10, 1e20)
    n_h = jnp.clip(y[2], 1e-10, 1e20)
    p = n_h + p0

    auger = ka * (n_e * p**2 + p * n_e**2)
    bimolecular = kb * n_e * p
    trapping = kt * n_e * (NT - n_t)
    detrapping = kdt * n_t
    depopulation = kdp * n_t * p

    dne_dt = -bimolecular - auger - trapping + detrapping
    dnt_dt = trapping - depopulation - detrapping
    dnh_dt = -bimolecular - auger - depopulation

    # Stop the trap population growing beyond the trap density
    dnt_dt = jnp.where(n_t <= NT, dnt_dt, -jnp.abs(dnt_dt))

    return jnp.stack([dne_dt, dnt_dt, dnh_dt])


@jax.jit
def solve_BTD(t, ka, kt, kb, kdt, kdp, NT, p0, N0):
    """
    Solve the BTD model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    ka : float
        Auger rate constant (cm^6 ns^-1).
    kt : float
        Trapping rate constant (cm^3 ns^-1).
    kb : float
        Bimolecular rate constant (cm^3 ns^-1).
    kdt : float
        Detrapping rate constant, trap to conduction band (ns^-1).
    kdp : float
        Depopulation rate constant, trap to valence band (cm^3 ns^-1).
    NT : float
        Trap density (cm^-3).
    p0 : float
        Doping density (cm^-3).
    N0 : float
        Initial photo-excited carrier density (cm^-3).

    Returns
    -------
    diffrax.Solution
        ``sol.ys`` columns are [n_e, n_t, n_h].
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(BTD_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([N0, 0, N0]),
        args=jnp.array([ka, kt, kb, kdt, kdp, NT, p0]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        max_steps=10000,
        throw=False,
    )


@jax.jit
def TRPL_BTD(t, ka, kt, kb, kdt, kdp, NT, p0, N0, bkg):
    """
    TRPL signal for the BTD model, normalised to t = 0.

    Parameters
    ----------
    t, ka, kt, kb, kdt, kdp, NT, p0, N0
        See ``solve_BTD``.
    bkg : float
        Background added to the normalised signal.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n_e, n_h, n_t).
    """
    sol = solve_BTD(t, ka, kt, kb, kdt, kdp, NT, p0, N0)

    n = sol.ys[:, 0]
    nt = sol.ys[:, 1]
    p = sol.ys[:, 2]

    sig = n * (p + p0)
    sig = sig / sig[0]
    return jnp.log10(sig + bkg), n, p, nt


# =============================================================================
# DT model (dual trap)
# =============================================================================

def DT_Model(t, y, args):
    """
    Rate equations for the DT model (DOI: 10.1103/PRXEnergy.4.013001).

    A shallow trap captures (k_c) and re-emits (k_e) electrons, and a deep trap
    removes electrons non-radiatively (k_deep). Holes follow from charge
    neutrality, p = n + n_t.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, n_t]: free and shallow-trapped electrons.
    args : jnp.ndarray
        [k_c, k_deep, k_e, k_rad, k_aug].

    Returns
    -------
    jnp.ndarray
        [dn/dt, dn_t/dt].
    """
    n, nt = y
    k_c, k_deep, k_e, k_rad, k_aug = args

    p = n + nt

    radiative = -k_rad * n * p
    non_radiative = (-0.5 * k_aug * (n**2 * p + p**2 * n)
                     - k_c * n + k_e * nt
                     - k_deep * n)

    dn_dt = radiative + non_radiative
    dnt_dt = k_c * n - k_e * nt

    return jnp.array([dn_dt, dnt_dt])


@jax.jit
def solve_DT(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug):
    """
    Solve the DT model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    n_dens : float
        Initial electron density (cm^-3).
    k_c : float
        Shallow-trap capture rate constant (ns^-1).
    k_deep : float
        Deep-trap capture rate constant (ns^-1).
    k_e : float
        Shallow-trap emission rate constant (ns^-1).
    k_rad : float
        Radiative recombination rate constant (cm^3 ns^-1).
    k_aug : float
        Auger recombination rate constant (cm^6 ns^-1).

    Returns
    -------
    diffrax.Solution
        ``sol.ys`` columns are [n, n_t].
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(DT_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([n_dens, 0.0]),
        args=jnp.array([k_c, k_deep, k_e, k_rad, k_aug]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        max_steps=100000,
        throw=False,
    )


@jax.jit
def TRPL_DT(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug, p0, bkg):
    """
    TRPL signal for the DT model, normalised to t = 0.

    Parameters
    ----------
    t, n_dens, k_c, k_deep, k_e, k_rad, k_aug
        See ``solve_DT``.
    p0 : float
        Doping density (cm^-3).
    bkg : float
        Background added to the signal before normalisation.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n, p, n_t) with p = n + n_t.
    """
    sol = solve_DT(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug)

    n = sol.ys[:, 0]
    nt = sol.ys[:, 1]

    sig = k_rad * n * (n + p0)
    sig = sig + bkg
    sig = sig / sig[0]
    return jnp.log10(sig), n, n + nt, nt


# =============================================================================
# DTShallowVar model (dual trap, variable shallow-trap density)
# =============================================================================

def DTShallowVar_Model(t, y, args):
    """
    Rate equations for the DTShallowVar model.

    Trap 1 (shallow) captures electrons with rate beta_n_t1 * n * (N_t1 - n_t1)
    and re-emits them (detrapping) but does not recombine with holes. Trap 2
    (deep) removes electrons at rate beta_n_t2 * n. Holes follow from charge
    neutrality, p = n + n_t1.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, n_t1].
    args : jnp.ndarray
        [krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2].

    Returns
    -------
    jnp.ndarray
        [dn/dt, dn_t1/dt].
    """
    n, nt1 = y
    krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2 = args

    # Keep densities within a physical range
    n = jnp.clip(n, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, N_t1)

    p = n + nt1

    dn_dt = (-krad * n * p
             - beta_n_t1 * n * (N_t1 - nt1) + e_n_t1 * nt1
             - beta_n_t2 * n)
    dnt1_dt = beta_n_t1 * n * (N_t1 - nt1) - e_n_t1 * nt1

    return jnp.stack([dn_dt, dnt1_dt])


@jax.jit
def solve_DTShallowVar(t, krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2, N0):
    """
    Solve the DTShallowVar model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    krad : float
        Radiative recombination rate constant (cm^3 ns^-1).
    beta_n_t1 : float
        Shallow-trap electron capture coefficient (cm^3 ns^-1).
    e_n_t1 : float
        Shallow-trap electron emission (detrapping) rate (ns^-1).
    N_t1 : float
        Shallow-trap density (cm^-3).
    beta_n_t2 : float
        Deep-trap electron capture rate (ns^-1).
    N0 : float
        Initial photo-excited carrier density (cm^-3).

    Returns
    -------
    diffrax.Solution
        ``sol.ys`` columns are [n, n_t1].
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(DTShallowVar_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([N0, 0]),
        args=jnp.array([krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        max_steps=10000,
        throw=False,
    )


@jax.jit
def TRPL_DTShallowVar(t, krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2, N0, bkg):
    """
    TRPL signal for the DTShallowVar model, normalised to t = 0.

    Parameters
    ----------
    t, krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2, N0
        See ``solve_DTShallowVar``.
    bkg : float
        Background added to the normalised signal.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n, p, n_t1).
    """
    sol = solve_DTShallowVar(t, krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2, N0)

    n = sol.ys[:, 0]
    nt1 = sol.ys[:, 1]
    p = n + nt1

    sig = n * p * krad
    sig = sig / sig[0]
    sig = jnp.log10(sig + bkg)

    return sig, n, p, nt1


# =============================================================================
# DTDeepVar model (dual trap, variable deep-trap density)
# =============================================================================

def DTDeepVar_Model(t, y, args):
    """
    Rate equations for the DTDeepVar model.

    Trap 1 (shallow) captures electrons at rate beta_n_t1 * n and re-emits them
    (detrapping) but does not recombine with holes. Trap 2 (deep) captures
    electrons with rate beta_n_t2 * n * (N_t2 - n_t2) and recombines them with
    holes (depopulation) at rate beta_p_t2 * p * n_t2.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, n_t1, n_t2, p].
    args : jnp.ndarray
        [krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2].

    Returns
    -------
    jnp.ndarray
        [dn/dt, dn_t1/dt, dn_t2/dt, dp/dt].
    """
    n, nt1, nt2, p = y
    krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2 = args

    # Keep densities within a physical range
    n = jnp.clip(n, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, 1e18)
    nt2 = jnp.clip(nt2, 1e-10, 1e18)
    p = jnp.clip(p, 1e-10, 1e20)

    dn_dt = (-krad * n * p
             - beta_n_t1 * n + e_n_t1 * nt1
             - beta_n_t2 * n * (N_t2 - nt2))
    dnt1_dt = beta_n_t1 * n - e_n_t1 * nt1
    dnt2_dt = beta_n_t2 * n * (N_t2 - nt2) - beta_p_t2 * p * nt2
    dp_dt = -krad * n * p - beta_p_t2 * p * nt2

    return jnp.stack([dn_dt, dnt1_dt, dnt2_dt, dp_dt])


@jax.jit
def solve_DTDeepVar(t, krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2, N0):
    """
    Solve the DTDeepVar model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    krad : float
        Radiative recombination rate constant (cm^3 ns^-1).
    beta_n_t1 : float
        Shallow-trap electron capture rate (ns^-1).
    e_n_t1 : float
        Shallow-trap electron emission (detrapping) rate (ns^-1).
    beta_n_t2 : float
        Deep-trap electron capture coefficient (cm^3 ns^-1).
    beta_p_t2 : float
        Deep-trap hole capture (depopulation) coefficient (cm^3 ns^-1).
    N_t2 : float
        Deep-trap density (cm^-3).
    N0 : float
        Initial photo-excited carrier density (cm^-3).

    Returns
    -------
    diffrax.Solution
        ``sol.ys`` columns are [n, n_t1, n_t2, p].
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(DTDeepVar_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([N0, 0, 0, N0]),
        args=jnp.array([krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        max_steps=10000,
        throw=False,
    )


@jax.jit
def TRPL_DTDeepVar(t, krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2, N0, bkg):
    """
    TRPL signal for the DTDeepVar model, normalised to t = 0.

    Parameters
    ----------
    t, krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2, N0
        See ``solve_DTDeepVar``.
    bkg : float
        Background added to the signal before normalisation.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n, p, n_t1, n_t2).
    """
    sol = solve_DTDeepVar(t, krad, beta_n_t1, e_n_t1, beta_n_t2, beta_p_t2, N_t2, N0)

    n = sol.ys[:, 0]
    nt1 = sol.ys[:, 1]
    nt2 = sol.ys[:, 2]
    p = sol.ys[:, 3]

    sig = n * p * krad
    sig = sig + bkg
    sig = sig / sig[0]
    sig = jnp.log10(sig)

    return sig, n, p, nt1, nt2


# =============================================================================
# ShallowTrapVar model (single shallow trap, variable density)
# =============================================================================

def ShallowTrapVar_Model(t, y, args):
    """
    Rate equations for a single shallow trap with finite density.

        dn/dt   = -k_rad*n*p - k_aug*n^2*p - k_c*n*(NT - n_t) + k_e*n_t
        dn_t/dt =  k_c*n*(NT - n_t) - k_e*n_t

    with p = n + n_t from charge neutrality.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, n_t].
    args : jnp.ndarray
        [k_c, k_e, k_rad, k_aug, NT].

    Returns
    -------
    jnp.ndarray
        [dn/dt, dn_t/dt].
    """
    n, nt = y
    k_c, k_e, k_rad, k_aug, NT = args

    p = n + nt

    trapping = k_c * n * (NT - nt)
    detrapping = k_e * nt
    radiative = k_rad * n * p
    auger = k_aug * n * n * p

    dn_dt = -radiative - auger - trapping + detrapping
    dnt_dt = trapping - detrapping

    return jnp.array([dn_dt, dnt_dt])


@jax.jit
def solve_ShallowTrapVar(t, n_dens, k_c, k_e, k_rad, k_aug, NT):
    """
    Solve the ShallowTrapVar model.

    Parameters
    ----------
    t : jnp.ndarray
        Time array (ns).
    n_dens : float
        Initial electron density (cm^-3).
    k_c : float
        Capture rate constant (cm^3 ns^-1).
    k_e : float
        Emission rate constant (ns^-1).
    k_rad : float
        Radiative recombination rate constant (cm^3 ns^-1).
    k_aug : float
        Auger recombination rate constant (cm^6 ns^-1).
    NT : float
        Trap density (cm^-3).

    Returns
    -------
    diffrax.Solution
        ``sol.ys`` columns are [n, n_t].
    """
    return diffrax.diffeqsolve(
        diffrax.ODETerm(ShallowTrapVar_Model),
        diffrax.Kvaerno5(),
        t0=t[0],
        t1=t[-1],
        dt0=t[1] - t[0],
        y0=jnp.array([n_dens, 0.0]),
        args=jnp.array([k_c, k_e, k_rad, k_aug, NT]),
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        max_steps=100000,
        throw=False,
    )


@jax.jit
def TRPL_ShallowTrapVar(t, n_dens, k_c, k_e, k_rad, k_aug, NT, p0, bkg):
    """
    TRPL signal for the ShallowTrapVar model, normalised to t = 0.

    Parameters
    ----------
    t, n_dens, k_c, k_e, k_rad, k_aug, NT
        See ``solve_ShallowTrapVar``.
    p0 : float
        Doping density (cm^-3).
    bkg : float
        Background added to the normalised signal.

    Returns
    -------
    tuple of jnp.ndarray
        (log10 signal, n, p, n_t) with p = n + n_t (photo-generated holes).
    """
    sol = solve_ShallowTrapVar(t, n_dens, k_c, k_e, k_rad, k_aug, NT)

    n = sol.ys[:, 0]
    nt = sol.ys[:, 1]
    p = n + nt

    sig = n * (p + p0) * k_rad
    sig = jnp.maximum(sig, 1e-50)
    sig = sig / sig[0]
    return jnp.log10(sig + bkg), n, p, nt


# =============================================================================
# FullREM model (two-trap Shockley-Read-Hall, rate equations only)
# =============================================================================

def FullREM_Model(t, y, args):
    """
    Rate equations for the full two-trap SRH model.

    Parameters
    ----------
    t : float
        Time (unused, required by Diffrax).
    y : jnp.ndarray
        State vector [n, p, n_t1, n_t2].
    args : jnp.ndarray
        [krad, ni,
         beta_n_t1, beta_p_t1, e_n_t1, e_p_t1, N_t1,
         beta_n_t2, beta_p_t2, e_n_t2, e_p_t2, N_t2].

    Returns
    -------
    jnp.ndarray
        [dn/dt, dp/dt, dn_t1/dt, dn_t2/dt].
    """
    n, p, nt1, nt2 = y
    (krad, ni,
     beta_n_t1, beta_p_t1, e_n_t1, e_p_t1, N_t1,
     beta_n_t2, beta_p_t2, e_n_t2, e_p_t2, N_t2) = args

    # Keep densities within a physical range
    n = jnp.clip(n, 1e-10, 1e20)
    p = jnp.clip(p, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, N_t1)
    nt2 = jnp.clip(nt2, 1e-10, N_t2)

    dn_dt = (-krad * (n * p - ni**2)
             - beta_n_t1 * n * (N_t1 - nt1) + e_n_t1 * nt1
             - beta_n_t2 * n * (N_t2 - nt2) + e_n_t2 * nt2)
    dp_dt = (-krad * (n * p - ni**2)
             - beta_p_t1 * p * nt1 + e_p_t1 * (N_t1 - nt1)
             - beta_p_t2 * p * nt2 + e_p_t2 * (N_t2 - nt2))
    dnt1_dt = (beta_n_t1 * n * (N_t1 - nt1)
               - beta_p_t1 * p * nt1
               - e_n_t1 * nt1 + e_p_t1 * (N_t1 - nt1))
    dnt2_dt = (beta_n_t2 * n * (N_t2 - nt2)
               - beta_p_t2 * p * nt2
               - e_n_t2 * nt2 + e_p_t2 * (N_t2 - nt2))

    return jnp.stack([dn_dt, dp_dt, dnt1_dt, dnt2_dt])


# =============================================================================
# Data utilities
# =============================================================================

@jax.jit
def standardise(x):
    """
    Standardise data to zero mean and unit standard deviation.

    Parameters
    ----------
    x : array_like
        Data to standardise.

    Returns
    -------
    tuple of jnp.ndarray
        (standardised data, mean, standard deviation).
    """
    mean = jnp.mean(x, keepdims=True)
    std = jnp.std(x, keepdims=True)
    return (x - mean) / std, mean, std


@jax.jit
def normalise(x):
    """
    Normalise data so the maximum along the last axis is 1.

    Parameters
    ----------
    x : array_like
        Data to normalise.

    Returns
    -------
    jnp.ndarray
        Normalised data.
    """
    return x / x.max(-1, keepdims=True)


def add_noise(signal, noise_amplitude=1e3):
    """
    Add Gaussian noise with a fixed absolute amplitude to a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (linear scale).
    noise_amplitude : float
        Standard deviation of the noise, in the same units as ``signal``.

    Returns
    -------
    np.ndarray
        Noisy signal.
    """
    noise = np.random.normal(0, noise_amplitude, size=signal.shape)
    return signal + noise


# =============================================================================
# Simulation and analysis helpers
# =============================================================================

K_B = 1.380649e-23         # Boltzmann constant (J K^-1)
Q_E = 1.602176634e-19      # elementary charge (C)


def colorFader(c1="blue", c2="red", factor=0.5):
    """Return the hex colour a fraction ``factor`` of the way from c1 to c2."""
    from matplotlib import colors as mcolors
    c1 = np.array(mcolors.to_rgb(c1))
    c2 = np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1 - factor) * c1 + factor * c2)


def diff_lifetime(time, PL, n=2):
    """
    Differential lifetime of a PL decay.

    Parameters
    ----------
    time : np.ndarray
        Time (ns).
    PL : np.ndarray
        PL signal (linear scale).
    n : float
        Order of the recombination process the lifetime refers to (2 for
        bimolecular emission).

    Returns
    -------
    np.ndarray
        Differential lifetime (s).
    """
    return (-np.gradient(np.log(PL), time * 1e-9) / n) ** -1


def diff_constant(time, PL, n0):
    """
    Differential recombination constant of a PL decay.

    The carrier density is estimated from the PL assuming PL ∝ n^2 and
    PL[0] ↔ n0.

    Parameters
    ----------
    time : np.ndarray
        Time (ns).
    PL : np.ndarray
        PL signal (linear scale).
    n0 : float
        Initial carrier density (cm^-3).

    Returns
    -------
    np.ndarray
        Differential constant (cm^3 s^-1).
    """
    proportionality = PL[0] / n0**2
    n_squared = PL / proportionality
    return -np.gradient(np.log(PL), time * 1e-9) / (2 * np.sqrt(n_squared))


def relative_QFLS(PL, n0, n0_max, eg, temperature=300):
    """
    Quasi-Fermi-level splitting relative to a reference carrier density.

    QFLS = eg + kT ln(n^2 / n_ref^2), https://doi.org/10.1002/aenm.202403279

    Parameters
    ----------
    PL : np.ndarray
        PL signal (linear scale).
    n0 : float
        Initial carrier density of this decay (cm^-3).
    n0_max : float or None
        Reference carrier density (cm^-3). Falls back to ``n0`` if not given.
    eg : float or None
        Band gap (eV) added as an offset. No offset if not given.
    temperature : float
        Temperature (K).

    Returns
    -------
    np.ndarray
        QFLS (eV).
    """
    n_squared = PL / (PL[0] / n0**2)
    n_ref = n0_max if n0_max else n0
    offset = eg if eg else 0
    return offset + K_B * temperature * np.log(n_squared / n_ref**2) / Q_E


def power_law(x):
    return 1 / x**2


def exp_decay(x, a):
    return np.exp(-a * x)
