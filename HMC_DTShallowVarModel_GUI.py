import numpy as np
import jax
import sys
import arviz as az
import numpyro
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from jax import jit
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.distributions import TruncatedNormal, Normal
import os

# Import your custom physics models
from globalfit_functions import *

# --- Configuration and model choices ---

MODEL_CONFIGS = {
    "DTShallowVar": {
        "function": TRPL_DTShallowVar,
        "parameters": [
            ("krad", -21.00, -20.00, -17.00),
            ("beta_n_t1", -18.50, -17.00, -16.00),
            ("e_n_t1", -6.50, -4.00, -1.00),
            ("N_t1", 14.00, 14.50, 17.00),
            ("beta_n_t2", -5.00, -2.00, 0.00),
            ("bkg", -12.00, -10.00, -1.00),
        ],
    },
}


def show_data_preview(data):
    plt.figure(figsize=(10, 6))
    for index in range(1, len(data)):
        plt.plot(data[0], data[index], label=f"Signal {index}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Signal intensity")
    plt.title("Raw TRPL data")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_standardised_data_preview(data):
    raw_signals = []
    for signal in data[1:]:
        normalized_signal = np.maximum(signal / signal[0], 1e-10)
        raw_signals.append(np.log10(normalized_signal))
    raw_signals = np.stack(raw_signals)
    global_mean = np.mean(raw_signals)
    global_std = np.std(raw_signals)
    if global_std < 1e-12:
        global_std = 1.0
    standardised_signals = (raw_signals - global_mean) / global_std

    plt.figure(figsize=(10, 6))
    for index, signal in enumerate(standardised_signals, start=1):
        plt.plot(data[0], signal, label=f"Signal {index}")
    plt.xscale("log")
    plt.xlabel("Time")
    plt.ylabel("Standardized signal")
    plt.title("Standardized TRPL data")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_configuration():
    root = tk.Tk()
    root.title("HMC TRPL configuration - DTShallowVar model")
    root.columnconfigure(1, weight=1)
    variables = {}
    parameter_rows = {}

    filename_var = tk.StringVar(value="Perovskite_TRPL_Data.npy")
    model_var = tk.StringVar(value="DTShallowVar")
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
    tk.Entry(root, textvariable=initial_density_var).grid(row=density_row, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

    parameter_row = density_row + 1
    parameter_frame = tk.LabelFrame(root, text="Parameters (log10 scale where applicable)")
    parameter_frame.grid(row=parameter_row, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
    headers = ["Use", "Parameter", "Lower", "Mean", "Upper"]
    for column, header in enumerate(headers):
        tk.Label(parameter_frame, text=header).grid(row=0, column=column, padx=5, pady=3)

    def rebuild_parameters(*_):
        for child in parameter_frame.winfo_children():
            if int(child.grid_info().get("row", 0)) > 0:
                child.destroy()
        parameter_rows.clear()
        config = MODEL_CONFIGS[model_var.get()]
        rows = [(name, lower, mean, upper) for name, lower, mean, upper in config["parameters"]]
        for row, (name, lower, mean, upper) in enumerate(rows, start=1):
            active = tk.BooleanVar(value=name not in {"ka", "k_aug", "p0", "bkg"})
            lower_var, mean_var, upper_var = (tk.StringVar(value=str(value)) for value in (lower, mean, upper))
            tk.Checkbutton(parameter_frame, variable=active).grid(row=row, column=0)
            tk.Label(parameter_frame, text=name).grid(row=row, column=1, sticky="w")
            for column, variable in enumerate((lower_var, mean_var, upper_var), start=2):
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
    tk.Button(root, text="Preview standardized data", command=preview_standardised).grid(row=button_row, column=1, padx=6, pady=8)
    tk.Checkbutton(root, text="Show plots after run", variable=show_plots_var).grid(row=button_row, column=2, padx=6, pady=8)
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


configuration = get_configuration()
filename = configuration["filename"]
model_name = configuration["model_name"]
num_devices = configuration["num_devices"]
num_samples = configuration["num_samples"]
rndint = configuration["rndint"]
accept_prob = configuration["accept_prob"]
warmups = configuration["warmups"]
max_tree_depth = configuration["max_tree_depth"]
selected_parameters = configuration["parameters"]
n0_list = configuration["n0_list"]

numpyro.set_host_device_count(num_devices)
jax.config.update("jax_enable_x64", True)
print(f"Configuration: model={model_name}, num_devices={num_devices}, num_samples={num_samples}")
print(f"Number of available devices for parallel chains: {jax.device_count()}")

# --- 1. Load Data ---
try:
    data = np.load(filename)
    print(f"Loaded {filename} successfully.")
except FileNotFoundError:
    print(f"COULD NOT FIND {filename} - ENSURE THE FILENAME IS CORRECT AND THE FILE IS IN THE CURRENT DIRECTORY.")
    sys.exit(1)


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

# --- 3. JIT Compile Functions ---
model_function = MODEL_CONFIGS[model_name]["function"]
runner = jit(model_function)


def run_selected_model(time_in, n0_val, physical_parameters):
    return runner(time_in, *physical_parameters[:5], n0_val, physical_parameters[5])[0]

# --- 4. Bayesian Model ---
def model(time_in, ydata=None):
    
    # --- Priors ---
    # N0 (Initial Carrier Densities) - Order: [Sig1, Sig2, Sig3, Sig4]
    density_values = jnp.array(n0_list)
    N0s_log = jnp.log10(density_values)
    decay_count = density_values.shape[0]

    # Factor to allow slight wiggle room in N0
    fac = numpyro.sample(
        "fac",
        TruncatedNormal(
            low   = 0.95 * jnp.ones(decay_count),
            high  = 1.05 * jnp.ones(decay_count),
            loc   = 1.00 * jnp.ones(decay_count),
            scale = 0.01 * jnp.ones(decay_count),
        ),
    )
    current_n0s = 10**(N0s_log * fac)

    # Model parameters are sampled in log10 space and masked after sampling.
    # An unchecked parameter therefore enters the forward model as exactly zero.
    model_parameters = MODEL_CONFIGS[model_name]["parameters"]
    parameter_values = selected_parameters
    theta_lower = jnp.array([parameter_values[name][1] for name, *_ in model_parameters])
    theta_mean = jnp.array([parameter_values[name][2] for name, *_ in model_parameters])
    theta_upper = jnp.array([parameter_values[name][3] for name, *_ in model_parameters])
    theta_scale = jnp.maximum((theta_upper - theta_lower) / 4.0, 1e-6)
    theta_mask = jnp.array([parameter_values[name][0] for name, *_ in model_parameters])
    theta_raw = numpyro.sample(
        "theta_raw",
        TruncatedNormal(low=theta_lower, high=theta_upper, loc=theta_mean, scale=theta_scale),
    )
    theta = numpyro.deterministic("theta", theta_raw * theta_mask)


    # Noise Prior (Shape: 4)
    noise = numpyro.sample(
        "noise",
        TruncatedNormal(
            low   = 0.01 * jnp.ones(decay_count),
            high  = 0.5 * jnp.ones(decay_count),
            loc   = 0.2 * jnp.ones(decay_count),
            scale = 0.1 * jnp.ones(decay_count),
        ),
    )

    # --- Vectorized Solver ---
    def get_signal_trace(n0_val):
        physical_parameters = (10**theta_raw) * theta_mask
        raw_output = run_selected_model(time_in, n0_val, physical_parameters)
        
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

kernel = NUTS(model, target_accept_prob=accept_prob, max_tree_depth=max_tree_depth)

samples = num_samples
mcmc = MCMC(
    kernel,
    num_warmup=warmups,
    num_samples=samples,
    num_chains=num_devices, 
    progress_bar=True,
)

print(f"Starting MCMC ({model_name} model)...")
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
filename = f"{filename[:-4]}_{model_name}_chains{num_devices}_WU{warmups}_SAM{samples}.nc"
az.to_netcdf(idata, filename)
print(f"Saved netcdf with priors to {filename}")


# --- 7. Save Prior, Posterior, and Diagnostic Information ---

parameter_definitions = {}
for index in range(len(n0_list)):
    parameter_definitions[f"fac[{index}]"] = (0.95, 1.05, 1.00)
    parameter_definitions[f"noise[{index}]"] = (0.01, 0.50, 0.20)
for index, (name, lower, mean, upper) in enumerate(MODEL_CONFIGS[model_name]["parameters"]):
    configured = selected_parameters[name]
    parameter_definitions[f"theta[{index}]"] = (configured[1], configured[3], configured[2])


def posterior_mode(values):
    """Return a stable histogram-based mode estimate for posterior samples."""
    values = np.asarray(values, dtype=float)
    if np.all(values == values[0]):
        return float(values[0])
    histogram, edges = np.histogram(values, bins="auto")
    mode_bin = int(np.argmax(histogram))
    return float((edges[mode_bin] + edges[mode_bin + 1]) / 2)


posterior_summary = az.summary(
    idata,
    var_names=["fac", "theta", "noise"],
    hdi_prob=0.94,
)
summary_rows = []
prior_sample_frames = []
posterior_sample_frames = []

for parameter, (lower, upper, starting_mean) in parameter_definitions.items():
    variable, index_text = parameter.rsplit("[", 1)
    index = int(index_text[:-1])
    prior_values = np.asarray(idata.prior[variable].values[..., index]).reshape(-1)
    posterior_values = np.asarray(idata.posterior[variable].values[..., index]).reshape(-1)
    diagnostics = posterior_summary.loc[parameter]

    summary_rows.append({
        "parameter": parameter,
        "model_parameter": (
            MODEL_CONFIGS[model_name]["parameters"][index][0]
            if variable == "theta" else parameter
        ),
        "prior_lower_bound": lower,
        "prior_upper_bound": upper,
        "prior_starting_mean": starting_mean,
        "prior_sample_mean": float(np.mean(prior_values)),
        "prior_sample_mode": posterior_mode(prior_values),
        "posterior_mean": float(diagnostics["mean"]),
        "posterior_mode": posterior_mode(posterior_values),
        "hdi_3%": float(diagnostics["hdi_3%"]),
        "hdi_97%": float(diagnostics["hdi_97%"]),
        "ess_bulk": float(diagnostics["ess_bulk"]),
        "ess_tail": float(diagnostics["ess_tail"]),
        "rhat": float(diagnostics["r_hat"]),
    })
    prior_sample_frames.append(pd.DataFrame({parameter: prior_values}))
    posterior_sample_frames.append(pd.DataFrame({parameter: posterior_values}))

summary_dataframe = pd.DataFrame(summary_rows)
prior_samples_dataframe = pd.concat(prior_sample_frames, axis=1)
posterior_samples_dataframe = pd.concat(posterior_sample_frames, axis=1)
excel_filename = f"{filename[:-3]}.xlsx"
with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    summary_dataframe.to_excel(writer, sheet_name="summary", index=False)
    prior_samples_dataframe.to_excel(writer, sheet_name="prior_samples", index=False)
    posterior_samples_dataframe.to_excel(writer, sheet_name="posterior_samples", index=False)
print(f"Saved prior, posterior, and diagnostics to {excel_filename}")


# --- 8. Optional Prior, Posterior, and Pair Plots ---

if configuration["show_plots"]:
    prior_samples_dataframe.hist(figsize=(12, 10), bins=30)
    plt.suptitle("Prior samples")
    plt.tight_layout()
    plt.show()

    plot_variables = ["fac", "theta", "noise"]
    az.plot_posterior(idata, var_names=plot_variables, hdi_prob=0.94)
    plt.tight_layout()
    plt.show()

    az.plot_pair(
        idata,
        var_names=["theta"],
        group="posterior",
        kind="scatter",
        scatter_kwargs={"alpha": 0.4},
    )
    plt.tight_layout()
    plt.show()