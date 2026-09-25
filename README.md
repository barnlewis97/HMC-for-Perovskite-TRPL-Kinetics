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
├── MCMC-Global-BTDModel-Yang_Stoi_Unified_nobkg_noAug.py   # Original BTD MCMC fitting script
├── HMC_GUI.py                                              # GUI: choose any model, sampler settings and priors
├── HMC_<Name>Model.py                                      # Console HMC fit for one model (prompts for settings)
├── globalfit_functions.py                                  # ODE models and TRPL signal functions
├── Perovskite_TRPL_Data.npy                                # Example TRPL data
├── HMC_Analysis.ipynb                                      # Analysis & results notebook
└── HMC_env.yml                                             # Conda environment specification
```

`<Name>` is one of `ABC`, `BTD`, `DT`, `DTShallowVar`, `DTDeepVar` or `ShallowTrapVar`.

---

## Physics Models (`globalfit_functions.py`)

Several charge carrier recombination models are implemented, all solved as systems of ODEs using [Diffrax](https://github.com/patrick-kidger/diffrax) (JAX-based adaptive ODE solver):

| Model | State Variables | Key Physics |
|---|---|---|
| `ABC` (`TRPL_ABC`, `TRPL_AB`) | n | First-order trapping, bimolecular and Auger recombination |
| **`BTD`** (active, `TRPL_BTD`) | n_e, n_t, n_h | Bimolecular, trapping, detrapping and depopulation with Auger |
| `DT` (`TRPL_DT`) | n, n_t | Dual trap: shallow trap with detrapping + deep non-radiative trap (DOI: 10.1103/PRXEnergy.4.013001) |
| `DTShallowVar` (`TRPL_DTShallowVar`) | n, n_t1 | Dual trap; shallow (detrapping-active) trap has variable density (N_T - n_T) |
| `DTDeepVar` (`TRPL_DTDeepVar`) | n, n_t1, n_t2, p | Dual trap; deep (depopulation-active) trap has variable density (N_T - n_T) |
| `ShallowTrapVar` (`TRPL_ShallowTrapVar`) | n, n_t | Single shallow trap with variable density + Auger |
| `FullREM` (rate equations only) | n, p, n_t1, n_t2 | Full two-trap SRH |

Each model has a `<Name>_Model` rate-equation function, a `solve_<Name>` solver and a `TRPL_<Name>` signal function. `TRPL_<Name>` returns `(log10 signal, carrier densities...)`, so the fitting scripts take element `[0]`.

The **BTD Model** (`TRPL_BTD`) is used in the main fitting script and includes:
- Radiative bimolecular recombination (rate `k_b`)
- Trap-mediated (SRH-like) non-radiative recombination (rate `k_t`)
- Deep trap non-radiative channel (rate `k_dt`)
- Trap density `N_T` and shallow trap emission `k_dp`

All models output the log₁₀ TRPL signal (proportional to `n × p`), normalised to the initial value, with a background `bkg` added.

The same `globalfit_functions.py` is used by the companion simulation repository [Perovskite-TRPL-Kinetics-Simulations](https://github.com/barnlewis97/Perovskite-TRPL-Kinetics-Simulations).

---

## Bayesian Model & MCMC (`MCMC-Global-BTDModel-Yang_Stoi_Unified_nobkg_noAug.py`)

### Data pipeline

1. Load TRPL decay data from a `.npy` file — shape `(5, N_time)`: one time axis and four signal channels at different fluences. The original script expects `Stoi_Decays.npy` (not included); the `HMC_<Name>` scripts default to the example `Perovskite_TRPL_Data.npy`.
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

`HMC_Analysis.ipynb` contains:

- Loading and visualising the posterior from the saved NetCDF
- Trace plots and R-hat convergence diagnostics
- Corner/pair plots of parameter correlations
- Posterior predictive checks (simulated decays overlaid on data)
- Summary statistics for all inferred parameters

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/barnlewis97/HMC-for-Perovskite-TRPL-Kinetics.git
cd HMC-for-Perovskite-TRPL-Kinetics
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
python MCMC-Global-BTDModel-Yang_Stoi_Unified_nobkg_noAug.py
```

This will:
1. Load `Stoi_Decays.npy` (edit `filename` in the script to use your own data)
2. Pre-process and standardise the data
3. Run 10 parallel MCMC chains (5000 warmup + 5000 samples each)
4. Print a summary table with R-hat and ESS diagnostics
5. Save the InferenceData object to a `.nc` file

Expected runtime depends heavily on hardware. On a modern multi-core CPU, expect several hours for 10 chains × 10,000 total steps with the stiff ODE solver.

### Fit any model with the GUI

```bash
python HMC_GUI.py
```

The window lets you pick the data file, the model (`ABC`, `BTD`, `DT`, `DTShallowVar`, `DTDeepVar` or `ShallowTrapVar`), the sampler settings, the nominal initial carrier densities and, for each parameter, whether to fit it and its prior bounds. Priors are in log₁₀ space with units of cm and ns, matched to the BTD parameter with the same role (e.g. bimolecular constants use the `kb` range). Auger constants, `p0` and `bkg` are unticked by default; unticked parameters are not sampled and are held at zero.

Results are written next to the data file as `<data>_<Model>_chains<C>_WU<W>_SAM<S>.nc` and `.xlsx`.

**NetCDF** (ArviZ `InferenceData`, open with `az.from_netcdf`):

| Group | Contents |
|---|---|
| `posterior` | One variable per fitted parameter (log₁₀), `fac` and `noise` per decay, and `fit` (noise-free standardised model curves) |
| `posterior.attrs` | Model, data file, sampler settings, runtime, divergences, fitted/fixed parameters and the full prior table (`parameters`, JSON) |
| `prior`, `prior_predictive` | 500 draws from the prior |
| `posterior_predictive` | Simulated observations from the posterior |
| `log_likelihood` | Pointwise log-likelihood (for `az.loo` / `az.compare`) |
| `sample_stats` | Divergences, acceptance rate, tree depth, leapfrog steps, energy |
| `observed_data` | The standardised log₁₀ decays that were fitted |
| `constant_data` | Raw decays and nominal N₀ values |

**Excel**:

| Sheet | Contents |
|---|---|
| `settings` | Model, data file, sampler settings, runtime, divergences, priors |
| `summary` | Per parameter: units, fitted/fixed, prior bounds and sample mean/mode, posterior mean/sd/median/mode, 94% HDI, linear-scale values, MCSE, ESS (bulk/tail), R-hat |
| `prior_samples`, `posterior_samples` | Every draw, with chain and draw columns |
| `sample_stats` | Per chain: divergences, mean acceptance rate, leapfrog steps, BFMI |
| `fit` | Data vs posterior mean fit and 94% HDI for each decay, as log₁₀(PL/PL₀) |

### Fit a model from the console

```bash
python HMC_DTDeepVarModel.py        # prompts for chains, samples, seed, acceptance probability, prior width
```

There is one console script per model (`HMC_<Name>Model.py`). They save only the `.nc` file.

A failed ODE solve for an extreme parameter draw returns NaN (via `EQX_ON_ERROR=nan`, set in `globalfit_functions.py`), so NUTS rejects that step instead of the run crashing.

### Analyse results

Open the notebook:

```bash
jupyter notebook HMC_Analysis.ipynb
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
| pandas / openpyxl | GUI `.xlsx` summaries |
| tkinter | `HMC_GUI.py` (ships with most Python installers) |

---

## Data Format

Input data (e.g. `Perovskite_TRPL_Data.npy`) is a NumPy array of shape `(5, N_time)`:

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
