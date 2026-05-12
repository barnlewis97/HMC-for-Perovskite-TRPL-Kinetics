# Bayesian TRPL Global Fitting with Hamiltonian Monte Carlo

A Bayesian inference framework for fitting **Time-Resolved Photoluminescence (TRPL)** decay data to physically-motivated charge carrier recombination models. Uses **Hamiltonian Monte Carlo (HMC/NUTS)** via NumPyro, with JAX-accelerated ODE solvers for fast, differentiable forward modelling.

This code was developed for the analysis of stoichiometric perovskite thin films and forms part of a thesis project on carrier dynamics in halide perovskites.

---

## Overview

TRPL spectroscopy probes how photogenerated charge carriers recombine in semiconductors over time. Extracting physically meaningful rate constants (radiative recombination, trap capture, trap density, etc.) requires fitting multi-exponential or ODE-based models to data — a problem well-suited to Bayesian inference, which naturally quantifies parameter uncertainty and correlations.

This repository implements a **global fit** across multiple excitation fluences simultaneously, using a shared set of physics parameters and fluence-dependent initial carrier densities. The Bayesian posterior is sampled using the **No-U-Turn Sampler (NUTS)**, a variant of HMC that eliminates the need to manually tune the number of leapfrog steps.

---

## Repository Structure

```
.
├── MCMC-Global-ExtendedModel-Yang_Stoi_Unified_nobkg_noAug.py   # Main MCMC fitting script
├── globalfit_functions.py                                         # ODE models and TRPL signal functions
├── Stoi_Decays.npy                                                # Experimental TRPL data (stoichiometric sample)
├── Thesis_HMC_BTD_Yang_Stoichiometric_5000WU_5000Sam_...ipynb    # Analysis & results notebook
└── HMC_env.yml                                                    # Conda environment specification
```

---

## Physics Models (`globalfit_functions.py`)

Several charge carrier recombination models are implemented, all solved as systems of ODEs using [Diffrax](https://github.com/patrick-kidger/diffrax) (JAX-based adaptive ODE solver):

| Model | State Variables | Key Physics |
|---|---|---|
| `Full_REM_Model` | n, p, n_t1, n_t2 | Radiative + two traps (full SRH) |
| `DualTrap_Model` | n, p, n_t1, n_t2 | Dual trap, simplified |
| `DualTrap_Model_v2` | n, n_t1 | Charge neutrality approximation |
| `DTDeepVar_Model` | n, n_t1 | Deep trap with variable density |
| `Manuel_Model` | n, n_t (trapped) | Single trap + Auger |
| `ShallowTrapVariable_Model` | n, n_t | Shallow trap, variable density |
| **`Extended_Model`** (active) | n, n_t | Trap + radiative + deep trap |

The **Extended Model** (`TRPL_Extended_Model`) is used in the main fitting script and includes:
- Radiative bimolecular recombination (rate `k_b`)
- Trap-mediated (SRH-like) non-radiative recombination (rate `k_t`)
- Deep trap non-radiative channel (rate `k_dt`)
- Trap density `N_T` and shallow trap emission `k_dp`

All models output the log₁₀ TRPL signal (proportional to `n × p × k_rad`), normalised to the initial value.

---

## Bayesian Model & MCMC (`MCMC-Global-ExtendedModel-Yang_Stoi_Unified_nobkg_noAug.py`)

### Data pipeline

1. Load TRPL decay data from `Stoi_Decays.npy` — shape `(5, N_time)`: one time axis and four signal channels at different fluences.
2. Divide each channel by its t=0 value, apply log₁₀, then standardise globally (zero mean, unit variance).

### Priors

| Parameter | Distribution | Range (log₁₀) |
|---|---|---|
| `k_t` (trap recombination) | Truncated Normal | −18.5 to −16.0 |
| `k_b` (bimolecular) | Truncated Normal | −21.0 to −17.0 |
| `k_dt` (deep trap) | Truncated Normal | −6.5 to −1.0 |
| `k_dp` (shallow emission) | Truncated Normal | −21.5 to −17.0 |
| `N_T` (trap density) | Truncated Normal | 10¹⁴ to 10¹⁷ cm⁻³ |
| `fac` (N₀ scale factor) | Truncated Normal | 0.95 to 1.05 per fluence |
| `noise` | Truncated Normal | 0.01 to 0.5 per channel |

Initial carrier densities `N₀` are fixed from the known fluences and allowed to vary by ±5% via the `fac` parameter to accommodate calibration uncertainty.

### Likelihood

A Normal likelihood is used on the standardised log-signal:

```
obs ~ Normal(model_prediction, noise)
```

All four fluence channels are fitted simultaneously (global fit), with the four ODE solves vectorised using `jax.vmap`.

### NUTS Sampler settings

```python
warmups   = 5000
samples   = 5000
chains    = 10
target_accept_prob = 0.85
max_tree_depth     = 8
```

### Output

Results are saved as a [NetCDF](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_netcdf.html) file via [ArviZ](https://python.arviz.org/), including posterior samples and prior predictive samples:

```
BTD_Yang_Stoichiometric_WU5000_SAM5000_nobkg_noAug_final.nc
```

This file can be loaded for posterior analysis, trace plots, pair plots, and predictive checks using ArviZ.

---

## Analysis Notebook

`Thesis_HMC_BTD_Yang_Stoichiometric_5000WU_5000Sam_nobkg_noAug_FINAL.ipynb` contains:

- Loading and visualising the posterior from the saved NetCDF
- Trace plots and R-hat convergence diagnostics
- Corner/pair plots of parameter correlations
- Posterior predictive checks (simulated decays overlaid on data)
- Summary statistics for all inferred parameters

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create the Conda environment

```bash
conda env create -f HMC_env.yml
conda activate MCMC_env
```

The environment includes JAX (CPU), NumPyro, Diffrax, Equinox, ArviZ, and all dependencies. See `HMC_env.yml` for the full package list.

> **Note:** The environment was built on Windows (prefix points to a Windows Anaconda installation). On Linux/macOS you may need to remove the `prefix:` line from `HMC_env.yml` before creating.

### 3. GPU / multi-device (optional)

For GPU acceleration, install the appropriate `jaxlib` CUDA wheel separately after creating the environment. The script uses `numpyro.set_host_device_count(20)` for multi-chain parallelism on CPU; adjust this to match your hardware.

---

## Usage

### Run the MCMC fit

```bash
python MCMC-Global-ExtendedModel-Yang_Stoi_Unified_nobkg_noAug.py
```

This will:
1. Load `Stoi_Decays.npy`
2. Pre-process and standardise the data
3. Run 10 parallel MCMC chains (5000 warmup + 5000 samples each)
4. Print a summary table with R-hat and ESS diagnostics
5. Save the InferenceData object to a `.nc` file

Expected runtime depends heavily on hardware. On a modern multi-core CPU, expect several hours for 10 chains × 10,000 total steps with the stiff ODE solver.

### Analyse results

Open the notebook:

```bash
jupyter notebook "Thesis_HMC_BTD_Yang_Stoichiometric_5000WU_5000Sam_nobkg_noAug_FINAL.ipynb"
```

---

## Dependencies

| Package | Role |
|---|---|
| [JAX](https://github.com/google/jax) | Autodiff + JIT compilation |
| [NumPyro](https://num.pyro.ai/) | Probabilistic programming / NUTS sampler |
| [Diffrax](https://github.com/patrick-kidger/diffrax) | JAX-based ODE solver (Kvaerno5, adaptive stepping) |
| [Equinox](https://github.com/patrick-kidger/equinox) | JAX neural network / pytree utilities |
| [ArviZ](https://python.arviz.org/) | MCMC diagnostics and visualisation |
| NumPy / SciPy | Data handling |
| Matplotlib / Seaborn | Plotting |

---

## Data Format

`Stoi_Decays.npy` is a NumPy array of shape `(5, N_time)`:

| Row | Content |
|---|---|
| `[0]` | Time axis (ns) |
| `[1]` | TRPL signal — fluence 1 (highest) |
| `[2]` | TRPL signal — fluence 2 |
| `[3]` | TRPL signal — fluence 3 |
| `[4]` | TRPL signal — fluence 4 (lowest) |

Signals are raw photon counts or intensity units; normalisation and log-transform are applied in the preprocessing step.

---

## Citation

If you use this code in your work, please cite this repository and the relevant thesis. A DOI or publication reference will be added here upon submission.

---

## Licence

MIT Licence — see `LICENSE` for details.
