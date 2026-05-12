import jax                     #numpy for CPU/GPU/TPU
import diffrax                 #jax-based numerical differential eq solver
import numpy as np
import equinox as eqx          #extension of jax
import jax.numpy as jnp        #jax numpy
import jax
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, SaveAt, Kvaerno5, PIDController
from functools import partial

jax.config.update("jax_enable_x64", True)
#%%
@jax.jit
def Full_REM_Model(t, y, args):
    """
    Rate Equation Model (REM) for charge carrier dynamics.
    
    Parameters:
        t: float - time (unused, but required for Diffrax compatibility)
        y: jnp.array - state vector [n, p, nt1, nt2]
        args: jnp.array - model parameters as described below

    Returns:
        dydt: jnp.array - derivatives [dn/dt, dp/dt, dnt1/dt, dnt2/dt]
    """
    n, p, nt1, nt2 = y
    (krad, ni,
     beta_n_t1, beta_p_t1, e_n_t1, e_p_t1, N_t1,
     beta_n_t2, beta_p_t2, e_n_t2, e_p_t2, N_t2) = args

    # Prevent nonphysical values
    n = jnp.clip(n, 1e-10, 1e20)
    p = jnp.clip(p, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, N_t1)
    nt2 = jnp.clip(nt2, 1e-10, N_t2)

    # Derivatives
    dn_dt = (-krad * (n * p - ni**2)
             - beta_n_t1 * n * (N_t1 - nt1) + e_n_t1 * nt1
             - beta_n_t2 * n * (N_t2 - nt2) + e_n_t2 * nt2) #Electron Recombination Rate

    dp_dt = (-krad * (n * p - ni**2)
             - beta_p_t1 * p * nt1 + e_p_t1 * (N_t1 - nt1)
             - beta_p_t2 * p * nt2 + e_p_t2 * (N_t2 - nt2)) #Hole recombination rate

    dnt1_dt = (beta_n_t1 * n * (N_t1 - nt1)
               - beta_p_t1 * p * nt1
               - e_n_t1 * nt1 + e_p_t1 * (N_t1 - nt1)) #Trap 1 - shallow non SRH active but variable trap dens

    dnt2_dt = (beta_n_t2 * n * (N_t2 - nt2)
               - beta_p_t2 * p * nt2
               - e_n_t2 * nt2 + e_p_t2 * (N_t2 - nt2))

    return jnp.stack([dn_dt, dp_dt, dnt1_dt, dnt2_dt])
#%%
@jax.jit
@eqx.filter_jit
def DualTrap_Model(t, y, args):
    """
    Rate Equation Model (REM) for charge carrier dynamics.
    
    Parameters:
        t: float - time (unused, but required for Diffrax compatibility)
        y: jnp.array - state vector [n, p, nt1, nt2]
        args: jnp.array - model parameters as described below

    Returns:
        dydt: jnp.array - derivatives [dn/dt, dp/dt, dnt1/dt, dnt2/dt]
    """
    n, p, nt1, nt2 = y
    
    (krad, 
     beta_n_t1, e_n_t1, N_t1,
     beta_n_t2, beta_p_t2) = args # ni,

    # Prevent nonphysical values
    n = jnp.clip(n, 1e-10, 1e20)
    p = jnp.clip(p, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, N_t1)
    nt2 = jnp.clip(nt2, 1e-10, N_t1)

    # Derivatives
    dn_dt = (-krad * (n * p) #- ni**2
             - beta_n_t1 * n * (N_t1 - nt1) + e_n_t1 * nt1
             - beta_n_t2 * n) #Electron Recombination Rate

    dp_dt = (-krad * (n * p) #- ni**2
             - beta_p_t2 * p * nt2) #Hole recombination rate

    dnt1_dt = (beta_n_t1 * n * (N_t1 - nt1)
               - e_n_t1 * nt1) #Trap 1 - shallow non SRH active but variable trap dens

    dnt2_dt = (beta_n_t2 * n
               - beta_p_t2 * p * nt2)

    return jnp.stack([dn_dt, dp_dt, dnt1_dt, dnt2_dt])
#%%

@jax.jit
@eqx.filter_jit
def solve_DualTrap_Model(t, krad,
                    beta_n_t1, e_n_t1, N_t1,
                    beta_n_t2, beta_p_t2,
                    N0):
    """
    Solve the REM system using JAX and Diffrax.

    Returns:
        sol: ODE solution object with .ys[:, 0] = n(t), etc.
    """
    # Pack parameters
    args = jnp.array([
        krad, 
        beta_n_t1, e_n_t1, N_t1,
        beta_n_t2, beta_p_t2
    ])

    # Initial conditions
    y0 = jnp.array([N0, N0, 0, 0])
    dt0 = t[1] - t[0]

    # Define model and solver
    terms = diffrax.ODETerm(DualTrap_Model)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)

    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
        throw = False
    )
    return sol
#%%

@jax.jit
@eqx.filter_jit
def TRPL_DualTrap_Model(t, krad,
                   beta_n_t1, e_n_t1, N_t1, 
                   beta_n_t2, beta_p_t2,
                   N0):
    """
    Calculate the TRPL signal (n*p) for the REM model.

    Returns:
        log10 signal
    """
    sol = solve_DualTrap_Model(
        t,
        krad,
        beta_n_t1, e_n_t1, N_t1,
        beta_n_t2, beta_p_t2, N0
    )

    n = sol.ys[:, 0]
    p = sol.ys[:, 1]
    N_t1 = sol.ys[:, 2]
    N_t2 = sol.ys[:, 3]
    sig = n * p * krad
    sig = jnp.log10(sig)# - jnp.log10(sig[0])  # Normalization
    # Clip signal to avoid log(0) or overflow
    # sig = jnp.clip(sig, 1e-40, 1e30)
    return sig, n, p, N_t1, N_t2  # Return log10 signal and carrier concentrations

@jax.jit
@eqx.filter_jit
def DualTrap_Model_v2(t, y, args):
    """
    Rate Equation Model (REM) for charge carrier dynamics with two traps - shallow has variable density.
    
    Parameters:
        t: float - time (unused, but required for Diffrax compatibility)
        y: jnp.array - state vector [n, nt1] (simplified to 2 variables)
        args: jnp.array - model parameters as described below

    Returns:
        dydt: jnp.array - derivatives [dn/dt, dnt1/dt]
    """
    n, nt1 = y  # Only unpack 2 values now
    
    (krad, 
     beta_n_t1, e_n_t1, N_t1,
     beta_n_t2) = args

    # Prevent nonphysical values
    n = jnp.clip(n, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, N_t1)
    
    # Calculate p from charge neutrality
    p = n + nt1  # Assuming charge neutrality: p = n + trapped electrons

    # Derivatives
    dn_dt = (-krad * (n * p)
             - beta_n_t1 * n * (N_t1 - nt1) + e_n_t1 * nt1
             - beta_n_t2 * n)  # Electron Recombination Rate

    dnt1_dt = (beta_n_t1 * n * (N_t1 - nt1)
               - e_n_t1 * nt1)  # Trap 1 - shallow non SRH active but variable trap dens

    return jnp.stack([dn_dt, dnt1_dt])
#%%

@jax.jit
@eqx.filter_jit
def solve_DualTrap_Model_v2(t, krad,
                    beta_n_t1, e_n_t1, N_t1,
                    beta_n_t2, N0):#, beta_p_t2,
    """
    Solve the REM system using JAX and Diffrax.

    Returns:
        sol: ODE solution object with .ys[:, 0] = n(t), etc.
    """
    # Pack parameters
    args = jnp.array([
        krad, 
        beta_n_t1, e_n_t1, N_t1,
        beta_n_t2
    ])

    # Initial conditions
    y0 = jnp.array([N0, 0])#N0, 0,
    dt0 = t[1] - t[0]

    # Define model and solver
    terms = diffrax.ODETerm(DualTrap_Model_v2)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)

    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
        throw = False
    )
    return sol
#%%

@jax.jit
@eqx.filter_jit
def TRPL_DualTrap_Model_v2(t, krad,
                   beta_n_t1, e_n_t1, N_t1,
                   beta_n_t2, N0, bkg):
    """
    Calculate the TRPL signal (n*p) for the DDual Trap model with variable shallow trap density.

    Returns:
        log10 signal, n, p, nt1
    """
    sol = solve_DualTrap_Model_v2(
        t,
        krad,
        beta_n_t1, e_n_t1, N_t1,
        beta_n_t2, N0
    )

    n = sol.ys[:, 0]
    nt1 = sol.ys[:, 1]
    p = n + nt1  # Calculate p from charge neutrality
    
    sig = n * p * krad
    sig = sig
    sig /= sig[0]
    sig = jnp.log10(sig+bkg)
    
    return sig, n, p, nt1
#%%
@jax.jit
@eqx.filter_jit
def DTDeepVar_Model(t, y, args):
    """
    Rate Equation Model (REM) for charge carrier dynamics - deep trap variable density.
    
    Parameters:
        t: float - time (unused, but required for Diffrax compatibility)
        y: jnp.array - state vector [n, nt1] (simplified to 2 variables)
        args: jnp.array - model parameters as described below

    Returns:
        dydt: jnp.array - derivatives [dn/dt, dnt1/dt]
    """
    n, nt1, nt2, p = y  # Only unpack 3 values now
    
    (krad, 
     beta_n_t1, e_n_t1, 
     beta_n_t2, beta_p_t2, N_t2) = args

    # Prevent nonphysical values
    n = jnp.clip(n, 1e-10, 1e20)
    nt1 = jnp.clip(nt1, 1e-10, 1e18)
    nt2 = jnp.clip(nt2, 1e-10, 1e18)
    p = jnp.clip(p, 1e-10, 1e20)

    # Derivatives
    dn_dt = (-krad * (n * p)
             - beta_n_t1 * n + e_n_t1 * nt1
             - beta_n_t2 * n * (N_t2 - nt2))  # Electron Recombination Rate

    dnt1_dt = (beta_n_t1 * n - e_n_t1 * nt1)  # Trap 1 - shallow non SRH active but variable trap dens
    
    dnt2_dt = beta_n_t2 * n * (N_t2 - nt2) - beta_p_t2 * p * nt2
    
    dp_dt = -krad * (n * p) - beta_p_t2 * p * nt2  # Hole recombination rate

    return jnp.stack([dn_dt, dnt1_dt, dnt2_dt, dp_dt])
#%%

@jax.jit
@eqx.filter_jit
def solve_DTDeepVar(t, krad,
                    beta_n_t1, e_n_t1,
                    beta_n_t2, beta_p_t2, N_t2, N0):#, beta_p_t2,
    """
    Solve the REM system using JAX and Diffrax.

    Returns:
        sol: ODE solution object with .ys[:, 0] = n(t), etc.
    """
    # Pack parameters
    args = jnp.array([
        krad, 
        beta_n_t1, e_n_t1,
        beta_n_t2, beta_p_t2, N_t2
    ])

    # Initial conditions
    y0 = jnp.array([N0, 0, 0, N0])#N0, 0,
    dt0 = t[1] - t[0]

    # Define model and solver
    terms = diffrax.ODETerm(DTDeepVar_Model)
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)

    sol = diffrax.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t[0],
        t1=t[-1],
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
        throw = False
    )
    return sol
#%%

@jax.jit
@eqx.filter_jit
def TRPL_DTDeepVar(t, krad,
                   beta_n_t1, e_n_t1,
                   beta_n_t2, beta_p_t2, N_t2, N0, bkg):
    """
    Calculate the TRPL signal (n*p) for the DDual Trap model with variable shallow trap density.

    Returns:
        log10 signal, n, p, nt1
    """
    sol = solve_DTDeepVar(
        t,
        krad,
        beta_n_t1, e_n_t1, 
        beta_n_t2, beta_p_t2, N_t2, N0
    )

    n = sol.ys[:, 0]
    nt1 = sol.ys[:, 1]
    nt2 = sol.ys[:, 2]
    p = sol.ys[:, 3]  # Calculate p from charge neutrality
    
    sig = n * p * krad
    sig = sig + bkg
    sig /= sig[0]
    
    sig = jnp.log10(sig)
    
    return sig, n, p, nt1, nt2

#%%
#Bayesian model for the BTDP model.
@jax.jit
@eqx.filter_jit
    
def Extended_Model(t, y, args):
    dne_dt, dnt_dt,  dnh_dt = y
    ka, kt, kb, kdt, kdp, NT, p0 = args
    
    # Add some numerical safeguards
    y0 = jnp.maximum(y[0], 1e-10)  # Prevent negative/zero values
    y1 = jnp.maximum(y[1], 1e-10)
    y2 = jnp.maximum(y[2], 1e-10)
    
    # Clip very large values
    y0 = jnp.minimum(y0, 1e20)
    y1 = jnp.minimum(y1, 1e20)
    y2 = jnp.minimum(y2, 1e20)
    
    A   = ka * (y0*(y2 + p0)**2 + (y2 + p0)*y0**2)
    B   = kb * y0 * (y2 + p0)
    T   = kt * y0 * (NT - y1)
    DT  = kdt * y1
    DP  = kdp * y1 * (y2 + p0)
    
    dne_dt  = - B - A - T + DT
    dnt_dt  =   T - DP - DT
    dnh_dt  = - B - A - DP
    
    # More robust constraint
    dnt_dt = jnp.where(y1 <= NT, dnt_dt, -jnp.abs(dnt_dt))
    
    return jnp.stack([dne_dt, dnt_dt, dnh_dt])


#JIT compiled function to solve the ODE
@jax.jit #JIT = 'Just in time' - takes python code and translates to computer 1s and 0s
@eqx.filter_jit
def solve_TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0):
    """
    Formerly solve_TRPL_Extended_Model_LOW
    Solve the ODEs for the Extended_Model model.
    Solves for electron concentration in conduction band
    Solves for electron concentration in traps
    Solves for hole concentration in valence band

    Parameters
    ----------
    ka: float
        k_A Auger rate constant (cm^6 ns^-1).

    kt: float
        k_T trapping rate constant (cm^3 ns^-1).

    kb: float
        k_B bimolecular rate constant (cm^3 ns^-1).

    kdt: float
        k_Dt detrapping rate constant (ns^-1) (trap to conduction band).

    kdp: float
        k_Dp depopulation rate constant (cm^3 ns^-1) (trap to valence band).
    
    NT: float
        Trap density (cm^-3).

    p0: float
        Doping density (cm^-3).

    N0: float
        Initial electron concentration (cm^-3).

    NTp: float
        Initial density of carriers in traps (cm^-3).

    N0h: float
        Initial hole concentration (cm^-3).
    
    Returns
    -------
    sol: array
        Solution to the ODEs.

    """

    #Define equations
    terms = diffrax.ODETerm(Extended_Model) #Ordinary Differential Term - Input Model here

    #t = jnp.arange(60, dtype=jnp.float64)/10 commented out as i will put time in as an argument
    #Start and end times
    t0 = t[0] #Originally 0
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([N0, 0, N0]) 
    dt0 = t[1]-t[0] #Originally 0.0002 - this may be more robust when feeding in new datasets

    #Define solver and times to save at
    solver = diffrax.Kvaerno5() #Choice of numerical solver
    saveat = diffrax.SaveAt(ts=t) #Defining time values to save at - set to t so all times

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6) #PID controller is used to dynamically adapt step sizes to match a desired error tolerance
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([ka, kt, kb, kdt, kdp, NT, p0]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=10000,
        throw = False
    )
    return sol
#%%
#Function to calculate the TRPL signal
@jax.jit
@eqx.filter_jit
def TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0, bkg): 
    """
    
    Calculate the TRPL signal for the BTD model with auger, accumulation included.

    Parameters
    ----------
    r: jnp.array
        
        ka: float
            k_A Auger rate constant (cm^6 ns^-1).

        kt: float
            k_T trapping rate constant (cm^3 ns^-1).
        
        kb: float
            k_B bimolecular rate constant (cm^3 ns^-1).
        
        kdt: float
            k_Dt detrapping rate constant (ns^-1) (trap to conduction band).

        kdp: float
            k_Dp depopulation rate constant (cm^3 ns^-1) (trap to valence band).

        NT: float
            Trap density (cm^-3).

        p0: float
            Doping density (cm^-3).
        
        bkr: float
            Background counts (counts).

    y0: float
        Initial TRPL counts (counts).
    
    N0: float
        Initial electron concentration (cm^-3).
    
    bkr: float
        Background counts (counts).


    Returns
    -------
    sig: array
        TRPL signal.
    
    """
    p0=0
    #Solve ODEs
    sol = solve_TRPL_Extended_Model(t, ka, kt, kb, kdt, kdp, NT, p0, N0) #formerly solve_TRPL_Extended_Model_LOW(...)

    # def body_fun(_, val):
    #     return solve_TRPL_Extended_Model(*r, N0 + val.ys[-1, 0], val.ys[-1, 1], N0 + val.ys[-1, 2] - r[-1])

    # sol = jax.lax.fori_loop(0, 10, body_fun, sol)
    n = sol.ys[:, 0].copy()
    # p = sol.ys[:, 2].copy()
    nt = sol.ys[:, 1].copy()
    p = sol.ys[:, 2].copy()  # Calculate p from charge neutrality
    #Calculate TRPL signal
    # sig = jnp.maximum(sig, 1e-30)  # Avoid log10(0) or log10(neg)
    sig = n * (p + p0)
    sig= sig/sig[0]
    
    return jnp.log10(sig+bkg)

#%%
#Standardise the data
@jax.jit
@eqx.filter_jit
def standardise(x):
    """
    Standardise the data to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    x:  numpy.ndarray
        The data to standardise.

    Returns
    -------
    x:  numpy.ndarray
        The standardised data.
    """
    mean =  jnp.mean(x, keepdims=True)
    std = jnp.std(x, keepdims=True)
    print(mean)
    print(std)
    stan_array = (x - mean) / std
    return stan_array, mean, std
@jax.jit
def normalise(x):
    """
    Standardise the data to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    x:  numpy.ndarray
        The data to standardise.

    Returns
    -------
    x:  numpy.ndarray
        The standardised data.
    """
    return x/x.max(-1, keepdims=True)
#%%
@jax.jit
@eqx.filter_jit
def TRPL_AB(t, n_0, k_A, k_B):
    def AB_rate_equations(t, n, args):
        """ Rate equation of the ABC model
        
        :param n_0: Initial concentration of the free electron
        :param k_A: SRH Rate Constant
        :param k_B: Bimolecular Rate Constant
        :param k_C: Auger Rate Constant""" 
        k_A, k_B = args
        dne_dt = - k_A*n - k_B*n**2
        
        return dne_dt
    #Solve the ordinary differential equations for the free electron concentration    
    #Define equations
    terms = diffrax.ODETerm(AB_rate_equations)

    
    #Start and end times
    t0 = t[0]
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([n_0])
    dt0 = 0.0002

    #Define solver and times to save at
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([k_A, k_B]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    
    #Calculate TRPL Signal
    
    signal = k_B*sol.ys**2 
    
    signal = jnp.log10(signal)
    
    return signal

#%%

@eqx.filter_jit
def TRPL_ABC(t, n_0, k_A, k_B, k_C):
    def ABC_rate_equations(t, n, args):
        """ Rate equation of the ABC model
        
        :param n_0: Initial concentration of the free electron
        :param k_A: SRH Rate Constant
        :param k_B: Bimolecular Rate Constant
        :param k_C: Auger Rate Constant""" 
        k_A, k_B, k_C = args
        dne_dt = - k_A*n - k_B*n**2 - k_C*n**3
        # Add some numerical safeguards
        n = jnp.maximum(n, 1e-5)  # Prevent negative/zero values
        
        
        return dne_dt
    #Solve the ordinary differential equations for the free electron concentration    
    #Define equations
    terms = diffrax.ODETerm(ABC_rate_equations)

    
    #Start and end times
    t0 = t[0]
    t1 = t[-1]

    #Initial conditions and initial time step
    y0 = jnp.array([n_0])
    dt0 = 0.0002

    #Define solver and times to save at
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)

    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([k_A, k_B, k_C]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000000000
    )
    
    #Calculate TRPL Signal
    
    signal = k_B*sol.ys**2 
    #Ensure no negative values
    
    signal = jnp.log10(signal)
    
    return signal

#%% Add noise

def add_noise(signal, noise_amplitude=1e3):
    """
    Add Gaussian noise with a fixed absolute amplitude to every trace.
    :param signal: input signal array (linear scale).
    :param noise_amplitude: std deviation of the Gaussian noise (in same units as signal).
    """
    noise = np.random.normal(0, noise_amplitude, size=signal.shape)
    return signal + noise

def Manuel_Model(t, y, args):
    """Model from DOI: 10.1103/PRXEnergy.4.013001
    Considers a shallow trap with a capture and emission rate and a deep trap where non-radiative recombination occurs.
    Considers radiative recombination and Auger recombination.

    Parameters:
    - n_dens: Electron density
    - nt: Trapped electron density
    - params: Model parameters (k_c, k_deep, k_e, k_rad, k_aug, etc.)
    
    Returns:
    - dn_dt: Time derivative of electron density
    - dnt_dt: Time derivative of trapped electron density
    """
    n_dens, nt = y

    k_c, k_deep, k_e, k_rad, k_aug = args
    
    p_dens = n_dens + nt
    
    R_rad = - k_rad*n_dens*p_dens  # Radiative recombination rate
    
    dnt_dt = k_c*n_dens - k_e*nt  # Hole density time derivative
    R_nr = -0.5*k_aug*((n_dens**2*p_dens)+(p_dens**2*n_dens)) - k_c*n_dens + k_e*nt - k_deep*n_dens  # Non-radiative recombination rate
    
    dn_dt = R_rad + R_nr  # Electron density time derivative

    return jnp.array([dn_dt, dnt_dt])

#JIT compiled function to solve the ODE
@jax.jit #JIT = 'Just in time' - takes python code and translates to computer 1s and 0s
def solve_Manuel_Model(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug):
    """
    Solve the ODEs for the Manuel model.
    Solves for electron concentration in conduction band
    Solves for electron concentration in traps

    Parameters
    ----------
    t: jnp.array
        Time array.

    n_dens: float
        Initial electron concentration (cm^-3).

    k_c: float
        Capture rate constant (cm^3 ns^-1).

    k_deep: float
        Deep trap rate constant (ns^-1).

    k_e: float
        Emission rate constant (ns^-1).

    k_rad: float
        Radiative recombination rate constant (cm^3 ns^-1).

    k_aug: float
        Auger recombination rate constant (cm^6 ns^-1).
    
    Returns
    -------
    sol: array
        Solution to the ODEs.

    """
    
    #Define equations
    terms = diffrax.ODETerm(Manuel_Model) #Ordinary Differential Term - Input Model here
    #Start and end times
    t0 = t[0] #Originally 0
    t1 = t[-1]
    #Initial conditions and initial time step
    y0 = jnp.array([n_dens, 0.0])  # Initial electron density and trapped electron density
    dt0 = t[1]-t[0] #Originally 0.0002 - this may be more robust when feeding in new datasets
    #Define solver and times to save at
    solver = diffrax.Kvaerno5() #Choice of numerical solver
    saveat = diffrax.SaveAt(ts=t) #Defining time values to save at - set to t so all times
    #Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6) #PID controller is used to dynamically adapt step sizes to match a desired error tolerance
    #Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args = jnp.array([k_c, k_deep, k_e, k_rad, k_aug]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
        throw = False
    )
    return sol

# Calculate TRPL signal
@jax.jit
@eqx.filter_jit
def TRPL_Manuel_Model(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug, p0, bkg):
    """
    Calculate the TRPL signal for the Manuel model.

    Parameters
    ----------
    t: jnp.array
        Time array.

    n_dens: float
        Initial electron concentration (cm^-3).

    k_c: float
        Capture rate constant (cm^3 ns^-1).

    k_deep: float
        Deep trap rate constant (ns^-1).

    k_e: float
        Emission rate constant (ns^-1).

    k_rad: float
        Radiative recombination rate constant (cm^3 ns^-1).

    k_aug: float
        Auger recombination rate constant (cm^6 ns^-1).
    
    p0: float
        Doping density (cm^-3).
    Returns
    -------
    sig: array
        TRPL signal.
    
    """
    
    #Solve ODEs
    sol = solve_Manuel_Model(t, n_dens, k_c, k_deep, k_e, k_rad, k_aug)
    
    #Calculate TRPL signal
    sig = k_rad*sol.ys[:, 0]*(sol.ys[:, 0]+p0)  # Product of electron density and trapped electron density
    
    sig = sig+bkg
    sig /= sig[0]  # Normalize to initial signal
    return jnp.log10(sig)  # Return the logarithm of the signal


#%%
def ShallowTrapVariable_Model(t, y, args):
    """Shallow trap model with variable trap density.
    
    Rate equations:
    - dn/dt = -k_rad*n*p - k_aug*n^2*p - k_c*n*(NT-nt) + k_e*nt
    - dnt/dt = k_c*n*(NT-nt) - k_e*nt
    
    where p = n + nt (total hole density, assuming charge neutrality)
    
    Parameters:
    - y: [n_dens, nt] - electron density and trapped electron density
    - args: [k_c, k_e, k_rad, k_aug, NT]
        - k_c: Capture rate (cm^3 ns^-1)
        - k_e: Emission rate (ns^-1)
        - k_rad: Radiative recombination (cm^3 ns^-1)
        - k_aug: Auger recombination (cm^6 ns^-1)
        - NT: Total trap density (cm^-3)
    
    Returns:
    - [dn_dt, dnt_dt]
    """
    n_dens, nt = y
    k_c, k_e, k_rad, k_aug, NT = args
    
    # Total hole density (charge neutrality)
    p_dens = n_dens + nt
    
    # Trapping and detrapping
    trapping = k_c * n_dens * (NT - nt)
    detrapping = k_e * nt
    
    # Recombination rates
    radiative = k_rad * n_dens * p_dens
    auger = k_aug * n_dens * n_dens * p_dens
    
    # Rate equations
    dn_dt = -radiative - auger - trapping + detrapping
    dnt_dt = trapping - detrapping
    
    return jnp.array([dn_dt, dnt_dt])


@jax.jit
def solve_ShallowTrapVariable(t, n_dens, k_c, k_e, k_rad, k_aug, NT):
    """
    Solve the ODEs for the shallow trap model.
    
    Parameters
    ----------
    t: jnp.array
        Time array (ns).
    n_dens: float
        Initial electron concentration (cm^-3).
    k_c: float
        Capture rate constant (cm^3 ns^-1).
    k_e: float
        Emission rate constant (ns^-1).
    k_rad: float
        Radiative recombination rate constant (cm^3 ns^-1).
    k_aug: float
        Auger recombination rate constant (cm^6 ns^-1).
    NT: float
        Total trap density (cm^-3).
    
    Returns
    -------
    sol: diffrax solution object
        Solution to the ODEs with ys[:, 0] = n(t), ys[:, 1] = nt(t)
    """
    
    # Define equations
    terms = diffrax.ODETerm(ShallowTrapVariable_Model)
    
    # Start and end times
    t0 = t[0]
    t1 = t[-1]
    
    # Initial conditions: [n0, nt0]
    y0 = jnp.array([n_dens, 0.0])
    
    # Initial time step
    dt0 = t[1] - t[0]
    
    # Define solver and times to save at
    solver = diffrax.Kvaerno5()
    saveat = diffrax.SaveAt(ts=t)
    
    # Controller for adaptive time stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-8)
    
    # Solve ODEs
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args=jnp.array([k_c, k_e, k_rad, k_aug, NT]),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000000
    )
    
    return sol


@jax.jit
@eqx.filter_jit
def TRPL_ShallowTrapVariable(t, n_dens, k_c, k_e, k_rad, k_aug, NT, p0):
    """
    Calculate the TRPL signal for the shallow trap model.
    
    TRPL signal = n * p_total where p_total = (n + nt) + p0
    
    Parameters
    ----------
    t: jnp.array
        Time array (ns).
    n_dens: float
        Initial electron concentration (cm^-3).
    k_c: float
        Capture rate constant (cm^3 ns^-1).
    k_e: float
        Emission rate constant (ns^-1).
    k_rad: float
        Radiative recombination rate constant (cm^3 ns^-1).
    k_aug: float
        Auger recombination rate constant (cm^6 ns^-1).
    NT: float
        Total trap density (cm^-3).
    p0: float
        Background doping/carrier density (cm^-3).
    
    Returns
    -------
    sig: array
        Log10 of TRPL signal.
    """
    
    # Solve ODEs
    sol = solve_ShallowTrapVariable(t, n_dens, k_c, k_e, k_rad, k_aug, NT)
    
    # Extract solutions
    n = sol.ys[:, 0]  # Free electron density
    nt = sol.ys[:, 1]  # Trapped electron density
    
    # Total hole density
    p_total = n + nt + p0
    
    # TRPL signal proportional to radiative recombination rate
    sig = n * p_total * k_rad
    
    # Avoid log of zero/negative values
    sig = jnp.maximum(sig, 1e-50)
    
    return jnp.log10(sig)