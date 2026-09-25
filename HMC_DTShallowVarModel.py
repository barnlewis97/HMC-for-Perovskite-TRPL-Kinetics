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
import os

# Import your custom physics models
from globalfit_functions import *

# --- Configuration ---

# Ask for inputs BEFORE JAX initializes
num_devices_input = input("Enter the number of devices for parallel chains (default is 4): ").strip()
num_devices = int(num_devices_input) if num_devices_input else 4

num_samples_input = input("Enter the number of samples for MCMC (default is 1000): ").strip()
num_samples = int(num_samples_input) if num_samples_input else 1000

print(f"Configuration: num_devices={num_devices}, num_samples={num_samples}")
rndint      = int(input("Choose a random integer: ").strip())
accept_prob = float(input('Choose and acceptance probability (between 0.7 and 0.99): '.strip()))
if accept_prob < 0.7 or accept_prob > 0.99:
    print("Warning: Acceptance probability should be between 0.7 and 0.99 for stable sampling, resetting to 0.85.")
    accept_prob = 0.85
theta_stddev = float(input('Choose a theta stddev: '.strip()))

#Tell NumPyro/JAX how many CPU host devices to create

numpyro.set_host_device_count(num_devices)

# Import JAX and set configurations

jax.config.update("jax_enable_x64", True)

# Now JAX will recognise the requested number of devices
num_devices_available = jax.device_count()
print(f"Number of available devices for parallel chains: {num_devices_available}")

# --- 1. Load Data ---
# Use the filename for your Stoi data here
filename = r'Perovskite_TRPL_Data.npy' 

try:
    data = np.load(filename)
    print(f"Loaded {filename} successfully.")
except FileNotFoundError:
    print(f"COULD NOT FIND {filename} - ENSURE THE FILENAME IS CORRECT AND THE FILE IS IN THE CURRENT DIRECTORY.")
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


# data[0] is time, data[1..4] are signals
time_axis = data[0]

# Extract Signals, Log Transform, and Normalize to t=0
raw_signals = []
for i in range(1, len(data)):
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

if jnp.isnan(ydata_jax).any():
    raise ValueError("Critical Error: ydata_jax contains NaNs. Check your data file or log transforms.")

#Ask user if they want to plot the standardised data
plot_standardised = input("Do you want to plot the standardised data? (y/n): ").strip().lower()
if plot_standardised == 'y':
    plt.figure(figsize=(10, 6))
    for i in range(ydata_jax.shape[0]):
        plt.plot(time_axis, ydata_jax[i], label=f'Standardised Signal {i+1}')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Standardised Signal Intensity (a.u.)')
    plt.title('Standardised Stoi Decay Signals')
    plt.legend()
    plt.grid(True)
    plt.show()
# --- 3. JIT Compile Functions ---
runner = jit(TRPL_DTShallowVar)

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

    # Physics Parameters (Theta), sampled in log10 space
    # Order: krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2
    # Held at zero: bkg
    theta = numpyro.sample(
        "theta",
        TruncatedNormal(
            low   = jnp.array([-21.00, -18.50, -6.50, 14.00, -5.00]),
            high  = jnp.array([-17.00, -16.00, -1.00, 17.00, 0.00]),
            loc   = jnp.array([-20.00, -17.00, -4.00, 14.50, -2.00]),
            scale = theta_stddev * jnp.ones(5),
        ),
    )

    krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2 = 10**theta
    bkg = 0.0


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
        raw_output = runner(time_in, krad, beta_n_t1, e_n_t1, N_t1, beta_n_t2, n0_val, bkg)[0]
        
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
key = PRNGKey(rndint)

kernel = NUTS(model, target_accept_prob=accept_prob, max_tree_depth=8)

warmups = num_samples
samples = num_samples   
mcmc = MCMC(
    kernel,
    num_warmup=warmups,
    num_samples=samples,
    num_chains=num_devices, 
    progress_bar=True,
)

print("Starting MCMC (DTShallowVar Model)...")
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
filename = f"{filename[:-4]}_DTShallowVar_chains{num_devices}_WU{warmups}_SAM{samples}.nc"
az.to_netcdf(idata, filename)
print(f"Saved netcdf with priors to {filename}")