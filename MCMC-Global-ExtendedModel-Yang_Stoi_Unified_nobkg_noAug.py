import numpy as np
import jax
import sys
import arviz as az
import numpyro
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import TruncatedNormal, Normal

# Import your custom physics models
from globalfit_functions import *

# --- Configuration ---
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(3) 

# --- 1. Load Data ---
# Use the filename for your Stoi data here
filename = r'Stoi_Decays.npy' 


try:
    data = np.load(filename)
    print(f"Loaded {filename} successfully.")
except FileNotFoundError:
    print(f"Could not find {filename}.")
    # stop the process
    sys.exit(1)

#Ask the user if they want to plot the raw data
plot_raw = input("Do you want to plot the raw data? (y/n): ").strip().lower()
if plot_raw == 'y':
    plt.figure(figsize=(10, 6))
    for i in range(1, len(data)):
        plt.plot(data[0], data[i], label=f'Signal {i}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Intensity (a.u.)')
    plt.title('Raw Stoi Decay Signals')
    plt.legend()
    plt.grid(True)
    plt.show()
# plt.plot(data[0], data[1], label='Signal 1')
# plt.plot(data[0], data[2], label='Signal 2')
# plt.plot(data[0], data[3], label='Signal 3')
# plt.plot(data[0], data[4], label='Signal 4')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()


# data[0] is time, data[1..4] are signals
time_axis = data[0]

# Extract Signals, Log Transform, and Normalize to t=0
raw_signals = []
for i in range(1, 5):
    # Log10 and normalize by the first point (t=0)
    # Added safety epsilon just in case
    val = data[i] / data[i][0]
    val = np.maximum(val, 1e-10) # Prevent log of zero or negative
    sig = np.log10(val)
    raw_signals.append(sig)

# Stack into shape (4, N_time)
raw_signals = np.stack(raw_signals) 

# --- 2. Standardise Data ---
global_mean = np.mean(raw_signals)
global_std = np.std(raw_signals)

# FIX: Safety check to prevent division by zero
if global_std < 1e-12:
    print("Warning: Data is flat (std ~ 0). Setting std=1.0 to avoid errors.")
    global_std = 1.0

print(f"Global Stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}")

# Standardise
ydata_standardised = (raw_signals - global_mean) / global_std
ydata_jax = jnp.array(ydata_standardised)
# plt.plot(time_axis, ydata_jax[0], label='Standardised Signal 1')
# plt.plot(time_axis, ydata_jax[1], label='Standardised Signal 2')
# plt.plot(time_axis, ydata_jax[2], label='Standardised Signal 3')
# plt.plot(time_axis, ydata_jax[3], label='Standardised Signal 4')
# plt.show()
# FIX: Fail fast if NaNs were created
if jnp.isnan(ydata_jax).any():
    raise ValueError("Critical Error: ydata_jax contains NaNs. Check your data file or log transforms.")

# --- 3. JIT Compile Functions ---
runner = jit(TRPL_Extended_Model)

# --- 4. Bayesian Model ---
def model(time_in, ydata=None):
    
    # --- Priors ---
    # N0 (Initial Carrier Densities) - Order: [Sig1, Sig2, Sig3, Sig4]
    n0_list = jnp.array([5.5e+16, 2.5193548393389264e+16, 7983870983473166.0, 4151612875643715.5]) 
    N0s_log = jnp.log10(n0_list)

    # Factor to allow slight wiggle room in N0
    fac = numpyro.sample(
        "fac",
        TruncatedNormal(
            low   = 0.95 * jnp.ones(4), 
            high  = 1.05 * jnp.ones(4),
            loc   = 1.00 * jnp.ones(4),
            scale = 0.01 * jnp.ones(4),
        ),
    )
    current_n0s = 10**(N0s_log * fac) # Shape (4,)

    # Physics Parameters (Theta)
    # Order: kt, kb, kdt, kdp, NT, bkg
    # FIX: Tightened 'bkg' lower bound from -20 to -10. 
    # 10^-20 is too small for double precision solvers and causes crashes.
    theta = numpyro.sample(
        "theta",
        TruncatedNormal(
            low   = jnp.array([-18.50, -21.00, -6.5, -21.5, 14]), 
            high  = jnp.array([-16.00, -17.0, -1.0, -17.00, 17.0]),
            loc   = jnp.array([-17.00, -20.00, -4.00, -19.00, 14.5]),
            scale = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ),
    )
    
    kt, kb, kdt, kdp, NT = theta[0], theta[1], theta[2], theta[3], theta[4]


    # Noise Prior (Shape: 4)
    noise = numpyro.sample(
        "noise",
        TruncatedNormal(
            low   = 0.01 * jnp.ones(4),
            high  = 0.5 * jnp.ones(4),
            loc   = 0.2 * jnp.ones(4),
            scale = 0.1 * jnp.ones(4),
        ),
    )

    # --- Vectorized Solver ---
    def get_signal_trace(n0_val):
        # Calls TRPL_Extended_Model
        raw_output = runner(time_in, 0, 10**kt, 10**kb, 10**kdt, 10**kdp, 10**NT, 0, n0_val, 0)
        
        # Normalize to t=0 (log scale)
        sig = raw_output - raw_output[0] 
        # Standardise
        sig_standardised = (sig - global_mean) / global_std
        return sig_standardised

    # Run all 4 simulations in parallel
    simulation = jax.vmap(get_signal_trace)(current_n0s)
    
    # --- Likelihood ---
    numpyro.sample("obs", Normal(simulation, noise[:, None]), obs=ydata)


# --- 5. Run MCMC ---
rndint      = 3 
accept_prob = 0.85
key = PRNGKey(rndint)

kernel = NUTS(model, target_accept_prob=accept_prob, max_tree_depth=8)

warmups = 50
samples = 50   
mcmc = MCMC(
    kernel,
    num_warmup=warmups,
    num_samples=samples,
    num_chains=3, # Set to desired chains
    progress_bar=True,
)

print("Starting MCMC (Extended Model)...")
mcmc.run(key, time_in=time_axis, ydata=ydata_jax)
mcmc.print_summary()


# --- 6. Save Data with Priors ---

# 1. Generate Prior Samples
# We use the same model but tell Predictive to sample from the priors
prior_predictive = Predictive(model, num_samples=500) # 500-1000 samples is usually enough
prior_predictions = prior_predictive(
    PRNGKey(8), 
    time_in=time_axis,
)

# 2. Convert to ArviZ InferenceData
# We include the MCMC object (posteriors) AND the prior predictions
idata = az.from_numpyro(
    mcmc,
    prior=prior_predictions,# Variables to include in prior_predictive
)

# 3. Save to NetCDF
filename = f"BTD_Yang_Stoichiometric_WU{warmups}_SAM{samples}_nobkg_noAug_final.nc"
az.to_netcdf(idata, filename)
print(f"Saved netcdf with priors to {filename}")