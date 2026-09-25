"""
GUI for fitting TRPL decays with any model in globalfit_functions.py using HMC (NUTS).

Run ``python HMC_GUI.py``, choose the data file, model, sampler settings and priors,
then press "Start MCMC". Results are written next to the data file:

    <data>_<Model>_chains<C>_WU<W>_SAM<S>.nc     ArviZ InferenceData (NetCDF)
    <data>_<Model>_chains<C>_WU<W>_SAM<S>.xlsx   settings, summary, samples, sampler stats, fit

Model parameters are sampled in log10 space. Unticked parameters are not sampled and
enter the forward model as exactly zero.
"""

import json
import sys
import time
from datetime import datetime

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import tkinter as tk
from jax import jit
from jax.random import PRNGKey
from numpyro.distributions import Normal, TruncatedNormal
from numpyro.infer import MCMC, NUTS, Predictive
from tkinter import filedialog, messagebox, ttk

from globalfit_functions import *


# =============================================================================
# Models and default priors
# =============================================================================

# Parameters held at zero unless ticked in the GUI
FIXED_BY_DEFAULT = {"ka", "k_aug", "p0", "bkg"}

# Each parameter is (name, lower, mean, upper, units) with the prior bounds and mean in log10
# space. "call" maps (runner, t, N0, physical parameters in listed order) onto the TRPL
# function's argument order.
MODEL_CONFIGS = {
    "ABC": {
        "function": TRPL_ABC,
        "call": lambda run, t, n0, p: run(t, n0, *p),
        "parameters": [
            ("k_A", -5.00, -2.00, 0.00, "ns^-1"),
            ("k_B", -21.00, -19.00, -17.00, "cm^3 ns^-1"),
            ("k_C", -40.00, -37.00, -34.00, "cm^6 ns^-1"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
    "BTD": {
        "function": TRPL_BTD,
        "call": lambda run, t, n0, p: run(t, *p[:7], n0, p[7]),
        "parameters": [
            ("ka", -30.00, -25.00, -18.00, "cm^6 ns^-1"),
            ("kt", -18.50, -17.00, -16.00, "cm^3 ns^-1"),
            ("kb", -21.00, -20.00, -17.00, "cm^3 ns^-1"),
            ("kdt", -6.50, -4.00, -1.00, "ns^-1"),
            ("kdp", -21.50, -19.00, -17.00, "cm^3 ns^-1"),
            ("NT", 14.00, 14.50, 17.00, "cm^-3"),
            ("p0", 12.00, 14.00, 18.00, "cm^-3"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
    "DT": {
        "function": TRPL_DT,
        "call": lambda run, t, n0, p: run(t, n0, *p),
        "parameters": [
            ("k_c", -5.00, -2.00, 0.00, "ns^-1"),
            ("k_deep", -5.00, -2.00, 0.00, "ns^-1"),
            ("k_e", -6.50, -4.00, -1.00, "ns^-1"),
            ("k_rad", -21.00, -20.00, -17.00, "cm^3 ns^-1"),
            ("k_aug", -40.00, -37.00, -34.00, "cm^6 ns^-1"),
            ("p0", 12.00, 14.00, 18.00, "cm^-3"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
    "DTShallowVar": {
        "function": TRPL_DTShallowVar,
        "call": lambda run, t, n0, p: run(t, *p[:5], n0, p[5]),
        "parameters": [
            ("krad", -21.00, -20.00, -17.00, "cm^3 ns^-1"),
            ("beta_n_t1", -18.50, -17.00, -16.00, "cm^3 ns^-1"),
            ("e_n_t1", -6.50, -4.00, -1.00, "ns^-1"),
            ("N_t1", 14.00, 14.50, 17.00, "cm^-3"),
            ("beta_n_t2", -5.00, -2.00, 0.00, "ns^-1"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
    "DTDeepVar": {
        "function": TRPL_DTDeepVar,
        "call": lambda run, t, n0, p: run(t, *p[:6], n0, p[6]),
        "parameters": [
            ("krad", -21.00, -20.00, -17.00, "cm^3 ns^-1"),
            ("beta_n_t1", -5.00, -2.00, 0.00, "ns^-1"),
            ("e_n_t1", -6.50, -4.00, -1.00, "ns^-1"),
            ("beta_n_t2", -18.50, -17.00, -16.00, "cm^3 ns^-1"),
            ("beta_p_t2", -21.50, -19.00, -17.00, "cm^3 ns^-1"),
            ("N_t2", 14.00, 14.50, 17.00, "cm^-3"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
    "ShallowTrapVar": {
        "function": TRPL_ShallowTrapVar,
        "call": lambda run, t, n0, p: run(t, n0, *p),
        "parameters": [
            ("k_c", -18.50, -17.00, -16.00, "cm^3 ns^-1"),
            ("k_e", -6.50, -4.00, -1.00, "ns^-1"),
            ("k_rad", -21.00, -20.00, -17.00, "cm^3 ns^-1"),
            ("k_aug", -40.00, -37.00, -34.00, "cm^6 ns^-1"),
            ("NT", 14.00, 14.50, 17.00, "cm^-3"),
            ("p0", 12.00, 14.00, 18.00, "cm^-3"),
            ("bkg", -12.00, -10.00, -1.00, "normalised PL"),
        ],
    },
}

# Priors on the nuisance parameters: (lower, upper, loc, scale), one per decay
FAC_PRIOR = (0.95, 1.05, 1.00, 0.01)      # multiplies log10(N0)
NOISE_PRIOR = (0.01, 0.50, 0.20, 0.10)    # likelihood std in standardised units

HDI_PROB = 0.94
PRIOR_DRAWS = 500


# =============================================================================
# Data handling
# =============================================================================

def preprocess(data):
    """
    Log-transform and standardise the decays.

    Each decay is normalised to its first point and log10-transformed, then all decays
    are standardised together (one global mean and standard deviation).

    Returns
    -------
    tuple
        (standardised log10 signals (n_decays, n_time), global mean, global std)
    """
    log_signals = np.stack([np.log10(np.maximum(signal / signal[0], 1e-10)) for signal in data[1:]])
    global_mean = float(np.mean(log_signals))
    global_std = float(np.std(log_signals))
    if global_std < 1e-12:
        print("Warning: Data is flat (std ~ 0). Setting std=1.0 to avoid errors.")
        global_std = 1.0
    return (log_signals - global_mean) / global_std, global_mean, global_std


def show_data_preview(data):
    plt.figure(figsize=(10, 6))
    for index in range(1, len(data)):
        plt.plot(data[0], data[index], label=f"Signal {index}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (ns)")
    plt.ylabel("Signal intensity")
    plt.title("Raw TRPL data")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_standardised_data_preview(data):
    standardised, _, _ = preprocess(data)
    plt.figure(figsize=(10, 6))
    for index, signal in enumerate(standardised, start=1):
        plt.plot(data[0], signal, label=f"Signal {index}")
    plt.xscale("log")
    plt.xlabel("Time (ns)")
    plt.ylabel("Standardised signal")
    plt.title("Standardised TRPL data")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# Configuration window
# =============================================================================

def get_configuration():
    """Show the configuration window and return the chosen settings as a dict."""
    root = tk.Tk()
    root.title("HMC TRPL configuration")
    root.columnconfigure(1, weight=1)
    variables = {}
    parameter_rows = {}

    filename_var = tk.StringVar(value="Perovskite_TRPL_Data.npy")
    model_var = tk.StringVar(value="BTD")
    tk.Label(root, text="Data file").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=filename_var, width=48).grid(row=0, column=1, sticky="ew", padx=6, pady=4)

    def choose_file():
        selected = filedialog.askopenfilename(filetypes=[("NumPy data", "*.npy"), ("All files", "*.*")])
        if selected:
            filename_var.set(selected)

    tk.Button(root, text="Browse...", command=choose_file).grid(row=0, column=2, padx=6, pady=4)
    tk.Label(root, text="Model").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    model_menu = ttk.Combobox(root, textvariable=model_var, values=list(MODEL_CONFIGS), state="readonly")
    model_menu.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

    numeric_defaults = [
        ("Chains / devices", "1"),
        ("Warmup steps", "1000"),
        ("Posterior samples", "1000"),
        ("Random seed", "8"),
        ("Acceptance probability", "0.85"),
        ("Maximum tree depth", "8"),
    ]
    for row, (label, value) in enumerate(numeric_defaults, start=2):
        variable = tk.StringVar(value=value)
        variables[label] = variable
        tk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        tk.Entry(root, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6, pady=4)

    initial_density_var = tk.StringVar(
        value="5.5e16, 2.5193548393389264e16, 7.983870983473166e15, 4.1516128756437155e15"
    )
    density_row = len(numeric_defaults) + 2
    tk.Label(root, text="Initial carrier densities").grid(row=density_row, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=initial_density_var).grid(
        row=density_row, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

    parameter_row = density_row + 1
    parameter_frame = tk.LabelFrame(root, text="Parameters (priors in log10 space)")
    parameter_frame.grid(row=parameter_row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    headers = ["Fit", "Parameter", "Units", "Lower", "Mean", "Upper"]
    for column, header in enumerate(headers):
        tk.Label(parameter_frame, text=header).grid(row=0, column=column, padx=5, pady=3)

    def rebuild_parameters(*_):
        for child in parameter_frame.winfo_children():
            if int(child.grid_info().get("row", 0)) > 0:
                child.destroy()
        parameter_rows.clear()
        for row, (name, lower, mean, upper, units) in enumerate(MODEL_CONFIGS[model_var.get()]["parameters"], start=1):
            active = tk.BooleanVar(value=name not in FIXED_BY_DEFAULT)
            lower_var, mean_var, upper_var = (tk.StringVar(value=str(value)) for value in (lower, mean, upper))
            tk.Checkbutton(parameter_frame, variable=active).grid(row=row, column=0)
            tk.Label(parameter_frame, text=name).grid(row=row, column=1, sticky="w")
            tk.Label(parameter_frame, text=units).grid(row=row, column=2, sticky="w")
            for column, variable in enumerate((lower_var, mean_var, upper_var), start=3):
                tk.Entry(parameter_frame, textvariable=variable, width=12).grid(row=row, column=column, padx=3, pady=2)
            parameter_rows[name] = (active, lower_var, mean_var, upper_var)

    model_menu.bind("<<ComboboxSelected>>", rebuild_parameters)
    rebuild_parameters()

    def preview():
        try:
            show_data_preview(np.load(filename_var.get()))
        except Exception as error:
            messagebox.showerror("Unable to load data", str(error))

    def preview_standardised():
        try:
            show_standardised_data_preview(np.load(filename_var.get()))
        except Exception as error:
            messagebox.showerror("Unable to load data", str(error))

    show_plots_var = tk.BooleanVar(value=True)
    button_row = parameter_row + 1
    tk.Button(root, text="Preview raw data", command=preview).grid(row=button_row, column=0, padx=6, pady=8)
    tk.Button(root, text="Preview standardised data", command=preview_standardised).grid(
        row=button_row, column=1, padx=6, pady=8)
    tk.Checkbutton(root, text="Show plots after run", variable=show_plots_var).grid(
        row=button_row, column=2, padx=6, pady=8)
    result = {}

    def submit():
        try:
            selected_data = np.load(filename_var.get())
            n0_values = [float(value.strip()) for value in initial_density_var.get().split(",") if value.strip()]
            expected_density_count = selected_data.shape[0] - 1
            if len(n0_values) != expected_density_count:
                raise ValueError(
                    f"Enter exactly {expected_density_count} initial carrier densities "
                    f"for the selected data ({expected_density_count} decays)."
                )
            if any(value <= 0 for value in n0_values):
                raise ValueError("Initial carrier densities must be positive.")
            result.update({
                "filename": filename_var.get(),
                "model_name": model_var.get(),
                "n0_list": n0_values,
                "num_devices": int(variables["Chains / devices"].get()),
                "warmups": int(variables["Warmup steps"].get()),
                "num_samples": int(variables["Posterior samples"].get()),
                "rndint": int(variables["Random seed"].get()),
                "accept_prob": float(variables["Acceptance probability"].get()),
                "max_tree_depth": int(variables["Maximum tree depth"].get()),
                "show_plots": show_plots_var.get(),
                "parameters": {
                    name: (active.get(), float(lower.get()), float(mean.get()), float(upper.get()))
                    for name, (active, lower, mean, upper) in parameter_rows.items()
                },
            })
            if not 0.7 <= result["accept_prob"] <= 0.99:
                raise ValueError("Acceptance probability must be between 0.7 and 0.99.")
            if result["num_devices"] < 1 or result["warmups"] < 1 or result["num_samples"] < 1:
                raise ValueError("Chains/devices, warmups, and samples must be positive.")
            if not any(active for active, *_ in result["parameters"].values()):
                raise ValueError("Tick at least one parameter to fit.")
            for name, (active, lower, mean, upper) in result["parameters"].items():
                if lower >= upper or not lower <= mean <= upper:
                    raise ValueError(f"Prior bounds are invalid for {name}.")
            root.destroy()
        except ValueError as error:
            messagebox.showerror("Invalid configuration", str(error))

    tk.Button(root, text="Start MCMC", command=submit).grid(row=button_row + 1, column=1, padx=6, pady=8)
    root.mainloop()
    if not result:
        raise SystemExit("Configuration cancelled.")
    return result


# =============================================================================
# Fitting
# =============================================================================

def build_model(configuration, global_mean, global_std):
    """Return the NumPyro model for the chosen TRPL model and priors."""
    model_config = MODEL_CONFIGS[configuration["model_name"]]
    selected = configuration["parameters"]
    runner = jit(model_config["function"])
    call = model_config["call"]
    n0s_log = jnp.log10(jnp.array(configuration["n0_list"]))
    decay_count = n0s_log.shape[0]

    def nuisance_prior(lower, upper, loc, scale):
        ones = jnp.ones(decay_count)
        return TruncatedNormal(low=lower * ones, high=upper * ones, loc=loc * ones, scale=scale * ones)

    def model(time_in, ydata=None):
        # Allow each initial carrier density to vary slightly around its nominal value
        fac = numpyro.sample("fac", nuisance_prior(*FAC_PRIOR))
        current_n0s = 10**(n0s_log * fac)

        # One sample site per fitted parameter (log10 space); unticked parameters are zero
        physical = []
        for name, *_ in model_config["parameters"]:
            active, lower, mean, upper = selected[name]
            if active:
                scale = max((upper - lower) / 4.0, 1e-6)
                log_value = numpyro.sample(name, TruncatedNormal(low=lower, high=upper, loc=mean, scale=scale))
                physical.append(10**log_value)
            else:
                physical.append(jnp.asarray(0.0))
        physical = jnp.stack(physical)

        noise = numpyro.sample("noise", nuisance_prior(*NOISE_PRIOR))

        def signal_trace(n0):
            log_signal = call(runner, time_in, n0, physical)[0]
            return (log_signal - log_signal[0] - global_mean) / global_std

        fit = numpyro.deterministic("fit", jax.vmap(signal_trace)(current_n0s))
        numpyro.sample("obs", Normal(fit, noise[:, None]), obs=ydata)

    return model


def run_fit(configuration):
    """Run the HMC fit described by ``configuration`` and save the .nc and .xlsx results."""
    filename = configuration["filename"]
    model_name = configuration["model_name"]
    num_chains = configuration["num_devices"]
    warmups = configuration["warmups"]
    samples = configuration["num_samples"]
    seed = configuration["rndint"]
    n0_list = configuration["n0_list"]
    selected = configuration["parameters"]
    model_parameters = MODEL_CONFIGS[model_name]["parameters"]
    fitted_names = [name for name, *_ in model_parameters if selected[name][0]]

    numpyro.set_host_device_count(num_chains)
    jax.config.update("jax_enable_x64", True)
    print(f"Configuration: model={model_name}, chains={num_chains}, warmup={warmups}, samples={samples}")
    print(f"Number of available devices for parallel chains: {jax.device_count()}")

    # --- Data ---
    try:
        data = np.load(filename)
        print(f"Loaded {filename} successfully.")
    except FileNotFoundError:
        print(f"COULD NOT FIND {filename} - ENSURE THE FILENAME IS CORRECT AND THE FILE IS IN THE CURRENT DIRECTORY.")
        sys.exit(1)
    time_axis = data[0]
    standardised, global_mean, global_std = preprocess(data)
    ydata = jnp.array(standardised)
    if jnp.isnan(ydata).any():
        raise ValueError("Critical Error: the standardised data contains NaNs. Check your data file.")
    print(f"Global Stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}")

    # --- MCMC ---
    model = build_model(configuration, global_mean, global_std)
    kernel = NUTS(model, target_accept_prob=configuration["accept_prob"],
                  max_tree_depth=configuration["max_tree_depth"])
    mcmc = MCMC(kernel, num_warmup=warmups, num_samples=samples, num_chains=num_chains, progress_bar=True)

    print(f"Starting MCMC ({model_name} model)...")
    start = time.time()
    mcmc.run(PRNGKey(seed), time_in=time_axis, ydata=ydata,
             extra_fields=("diverging", "accept_prob", "num_steps", "energy", "potential_energy"))
    runtime = time.time() - start
    mcmc.print_summary()

    # --- Prior and posterior predictive draws ---
    prior = Predictive(model, num_samples=PRIOR_DRAWS)(PRNGKey(seed + 1), time_in=time_axis)
    posterior_predictive = Predictive(model, posterior_samples=mcmc.get_samples())(
        PRNGKey(seed + 2), time_in=time_axis)

    decay_coords = np.arange(len(n0_list))
    idata = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive={"obs": posterior_predictive["obs"]},
        constant_data={
            "raw_data": data[1:],
            "n0_nominal": np.asarray(n0_list),
        },
        coords={"decay": decay_coords, "time": time_axis},
        dims={
            "fac": ["decay"], "noise": ["decay"], "fit": ["decay", "time"], "obs": ["decay", "time"],
            "raw_data": ["decay", "time"], "n0_nominal": ["decay"],
        },
    )

    divergences = int(idata.sample_stats["diverging"].sum())
    parameter_table = {
        name: {"fitted": bool(selected[name][0]), "lower": selected[name][1], "mean": selected[name][2],
               "upper": selected[name][3], "units": units}
        for name, _, _, _, units in model_parameters
    }
    settings = {
        "model": model_name,
        "data_file": str(filename),
        "created": datetime.now().isoformat(timespec="seconds"),
        "chains": num_chains,
        "warmup": warmups,
        "samples": samples,
        "seed": seed,
        "target_accept_prob": configuration["accept_prob"],
        "max_tree_depth": configuration["max_tree_depth"],
        "runtime_minutes": round(runtime / 60, 2),
        "divergences": divergences,
        "n0_nominal": ", ".join(f"{n0:.4e}" for n0 in n0_list),
        "fitted_parameters": ", ".join(fitted_names),
        "fixed_at_zero": ", ".join(name for name, *_ in model_parameters if not selected[name][0]),
        "parameter_space": "log10 (model parameters); linear (fac, noise)",
        "fac_prior": "TruncatedNormal(low=%g, high=%g, loc=%g, scale=%g) on log10(N0) multiplier" % FAC_PRIOR,
        "noise_prior": "TruncatedNormal(low=%g, high=%g, loc=%g, scale=%g), standardised units" % NOISE_PRIOR,
        "model_parameter_priors": "TruncatedNormal(lower, upper, mean, scale=(upper-lower)/4) in log10 space",
        "global_mean": global_mean,
        "global_std": global_std,
    }
    idata.posterior.attrs.update({**settings, "parameters": json.dumps(parameter_table)})

    base = f"{str(filename)[:-4]}_{model_name}_chains{num_chains}_WU{warmups}_SAM{samples}"
    az.to_netcdf(idata, f"{base}.nc")
    print(f"Saved InferenceData to {base}.nc")

    write_excel(f"{base}.xlsx", idata, settings, model_parameters, selected, fitted_names, time_axis)
    print(f"Saved settings, summary, samples, sampler stats and fit to {base}.xlsx")

    if configuration["show_plots"]:
        show_result_plots(idata, fitted_names, global_mean, global_std)
    return idata


# =============================================================================
# Output
# =============================================================================

def histogram_mode(values):
    """Histogram-based mode estimate for a set of samples."""
    values = np.asarray(values, dtype=float)
    if np.all(values == values[0]):
        return float(values[0])
    histogram, edges = np.histogram(values, bins="auto")
    mode_bin = int(np.argmax(histogram))
    return float((edges[mode_bin] + edges[mode_bin + 1]) / 2)


def flatten_samples(group, names):
    """Flatten (chain, draw, ...) samples into one column per scalar parameter."""
    columns = {}
    first = group[names[0]]
    columns["chain"] = np.repeat(first["chain"].values, first.sizes["draw"])
    columns["draw"] = np.tile(first["draw"].values, first.sizes["chain"])
    for name in names:
        values = group[name].values
        if values.ndim == 2:
            columns[name] = values.reshape(-1)
        else:
            for index in range(values.shape[-1]):
                columns[f"{name}[{index}]"] = values[..., index].reshape(-1)
    return pd.DataFrame(columns)


def write_excel(path, idata, settings, model_parameters, selected, fitted_names, time_axis):
    variable_names = fitted_names + ["fac", "noise"]
    summary = az.summary(idata, var_names=variable_names, hdi_prob=HDI_PROB)
    low_col, high_col = [column for column in summary.columns if column.startswith("hdi_")]
    prior_samples = flatten_samples(idata.prior, variable_names)
    posterior_samples = flatten_samples(idata.posterior, variable_names)

    def stats_row(label, prior_bounds, units, status, space):
        row = {"parameter": label, "units": units, "status": status, "space": space,
               "prior_lower": prior_bounds[0], "prior_mean": prior_bounds[1], "prior_upper": prior_bounds[2]}
        if status != "fitted":
            return row
        prior_values = prior_samples[label].values
        posterior_values = posterior_samples[label].values
        diagnostics = summary.loc[label]
        row.update({
            "prior_sample_mean": float(np.mean(prior_values)),
            "prior_sample_mode": histogram_mode(prior_values),
            "posterior_mean": float(diagnostics["mean"]),
            "posterior_sd": float(diagnostics["sd"]),
            "posterior_median": float(np.median(posterior_values)),
            "posterior_mode": histogram_mode(posterior_values),
            f"posterior_{low_col}": float(diagnostics[low_col]),
            f"posterior_{high_col}": float(diagnostics[high_col]),
            "mcse_mean": float(diagnostics["mcse_mean"]),
            "ess_bulk": float(diagnostics["ess_bulk"]),
            "ess_tail": float(diagnostics["ess_tail"]),
            "r_hat": float(diagnostics["r_hat"]),
        })
        if space == "log10":
            row.update({
                "posterior_mean_linear": 10**row["posterior_mean"],
                "posterior_mode_linear": 10**row["posterior_mode"],
                f"{low_col}_linear": 10**row[f"posterior_{low_col}"],
                f"{high_col}_linear": 10**row[f"posterior_{high_col}"],
            })
        return row

    rows = []
    for name, _, _, _, units in model_parameters:
        active, lower, mean, upper = selected[name]
        rows.append(stats_row(name, (lower, mean, upper), units, "fitted" if active else "fixed at 0", "log10"))
    decay_count = idata.posterior["fac"].sizes["decay"]
    for prefix, (lower, upper, loc, _), units in (("fac", FAC_PRIOR, ""), ("noise", NOISE_PRIOR, "standardised")):
        for index in range(decay_count):
            rows.append(stats_row(f"{prefix}[{index}]", (lower, loc, upper), units, "fitted", "linear"))
    summary_frame = pd.DataFrame(rows)

    stats = idata.sample_stats
    sample_stats_frame = pd.DataFrame({
        "chain": stats["chain"].values,
        "divergences": stats["diverging"].sum("draw").values,
        "mean_acceptance_rate": stats["acceptance_rate"].mean("draw").values,
        "mean_leapfrog_steps": stats["n_steps"].mean("draw").values,
        "max_leapfrog_steps": stats["n_steps"].max("draw").values,
        "bfmi": az.bfmi(idata),
    })

    # Data against the posterior fit, back-transformed to log10(PL / PL[0])
    fit = idata.posterior["fit"] * settings["global_std"] + settings["global_mean"]
    fit_mean = fit.mean(("chain", "draw")).values
    fit_hdi = az.hdi(fit, hdi_prob=HDI_PROB)["fit"].values
    data_log = idata.observed_data["obs"].values * settings["global_std"] + settings["global_mean"]
    fit_columns = {"time_ns": time_axis}
    for index in range(decay_count):
        fit_columns[f"data_log10_{index}"] = data_log[index]
        fit_columns[f"fit_mean_log10_{index}"] = fit_mean[index]
        fit_columns[f"fit_hdi_low_log10_{index}"] = fit_hdi[index, :, 0]
        fit_columns[f"fit_hdi_high_log10_{index}"] = fit_hdi[index, :, 1]

    settings_frame = pd.DataFrame({"setting": list(settings), "value": [str(v) for v in settings.values()]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        settings_frame.to_excel(writer, sheet_name="settings", index=False)
        summary_frame.to_excel(writer, sheet_name="summary", index=False)
        prior_samples.to_excel(writer, sheet_name="prior_samples", index=False)
        posterior_samples.to_excel(writer, sheet_name="posterior_samples", index=False)
        sample_stats_frame.to_excel(writer, sheet_name="sample_stats", index=False)
        pd.DataFrame(fit_columns).to_excel(writer, sheet_name="fit", index=False)


def show_result_plots(idata, fitted_names, global_mean, global_std):
    az.plot_trace(idata, var_names=fitted_names)
    plt.tight_layout()
    plt.show()

    az.plot_posterior(idata, var_names=fitted_names + ["fac", "noise"], hdi_prob=HDI_PROB)
    plt.tight_layout()
    plt.show()

    if len(fitted_names) > 1:
        az.plot_pair(idata, var_names=fitted_names, kind="scatter", divergences=True,
                     scatter_kwargs={"alpha": 0.4})
        plt.tight_layout()
        plt.show()

    time_axis = idata.posterior["time"].values
    fit = idata.posterior["fit"] * global_std + global_mean
    fit_hdi = az.hdi(fit, hdi_prob=HDI_PROB)["fit"].values
    data_log = idata.observed_data["obs"].values * global_std + global_mean
    plt.figure(figsize=(10, 6))
    for index in range(data_log.shape[0]):
        line, = plt.plot(time_axis, 10**data_log[index], ".", alpha=0.5, label=f"Data {index + 1}")
        plt.plot(time_axis, 10**fit.mean(("chain", "draw")).values[index], color=line.get_color())
        plt.fill_between(time_axis, 10**fit_hdi[index, :, 0], 10**fit_hdi[index, :, 1],
                         color=line.get_color(), alpha=0.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (ns)")
    plt.ylabel("PL / PL(0)")
    plt.title(f"Posterior mean fit with {int(HDI_PROB * 100)}% HDI")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_fit(get_configuration())
