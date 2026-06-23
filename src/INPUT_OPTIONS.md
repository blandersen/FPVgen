# FPVgen Input File Options Reference

FPVgen uses **TOML format** configuration files. The main entry point is `scripts/generate_table.py`, invoked as:

```
generate_table <config.toml> [--verbose]
```

---

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `config` | *(Positional)* Path to the TOML configuration file |
| `--verbose` / `-v` | Enable verbose debug logging |

---

## `[mechanism]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | string | **Required** | Path to the Cantera chemical mechanism file (e.g., `"gri30.yaml"`) |
| `transport_model` | string | **Required** | Transport model: `"unity-Lewis-number"`, `"mixture-averaged"`, or `"multicomponent"` |
| `prog_def` | dict | `{CO=1.0, H2=1.0, CO2=1.0, H2O=1.0}` | Progress variable definition — species names mapped to their weighting coefficients |

---

## `[conditions]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pressure` | float | `101325` | Operating pressure [Pa] |
| `initial_chi_st` | float | `1.0e-3` | Initial scalar dissipation rate at stoichiometric mixture fraction [1/s] |

---

## `[fuel_inlet]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `composition` | dict | **Required** | Species mole fractions for the fuel stream (e.g., `{ CH4 = 1.0 }`) |
| `temperature` | float | **Required** | Fuel inlet temperature [K] |

---

## `[oxidizer_inlet]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `composition` | dict | **Required** | Species mole fractions for the oxidizer stream (e.g., `{ O2 = 0.21, N2 = 0.79 }`) |
| `temperature` | float | **Required** | Oxidizer inlet temperature [K] |

---

## `[solver]` Section

FPVgen traces the S-curve using a **two-point temperature continuation** method. Rather than prescribing chi_st directly, Cantera's `TwoPointFlameControl` scheme pins the temperature at two interior grid points (the "control points") and lets the strain rate — and therefore chi_st — be a free variable. Each iteration lowers the control point temperatures by `delta_T`, driving the solver up the upper branch toward extinction, through the turning point, and down the unstable branch.

### Output and logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | string | `"flamelet_results"` | Directory where `solutions.h5`, plots, and any FlameMaster files are written. Created automatically if it does not exist. |
| `loglevel` | int | `0` | Cantera solver verbosity. `0`=silent, `1`=basic iteration counts, `2`=detailed convergence info, `3`=full Jacobian diagnostics. Increase only for debugging; higher levels substantially slow output. |
| `write_FlameMaster` | boolean | `false` | If true, writes each converged solution as a FlameMaster-format text file in addition to the HDF5. Useful for interoperability with other flamelet solvers. |
| `create_plots` | boolean | `true` | Generate S-curve, temperature profile, progress variable profile, and strain/chi_st diagnostic plots after the run completes. |
| `strain_chi_st_model_param_file` | string | `null` | Path to a JSON file containing pre-fitted slope and intercept for the log-linear `strain_rate` vs. `chi_st` model. If omitted, this mapping is fitted fresh from the computed solutions. |

### Counterflow domain

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width_ratio` | float | `10.0` | Sets the physical domain size as `width = width_ratio × flame_thickness`. The flame thickness is estimated from chi_st at initialization and remeasured between steps if `width_change_enable=true`. A value of 10 gives substantial buffer on both sides of the reaction zone; values much below 5 can cause the flame to feel the boundaries. |
| `width_change_enable` | boolean | `false` | If `false`, the domain is fixed at the size computed from the initial chi_st. If `true`, the domain is resized between steps to track the changing flame thickness as chi_st increases toward extinction. Enabling this avoids the flame becoming very thin relative to the domain on the upper branch, but each resize requires a re-solve and increases compute time. |
| `width_change_max` | float | `0.2` | When `width_change_enable=true`, the domain width cannot change by more than this fraction in a single step (e.g., `0.2` = ±20%). Prevents the grid from jumping to an extreme size in one shot, which can cause solver divergence. |
| `width_change_min` | float | `0.05` | When `width_change_enable=true`, a resize is skipped if the required fractional change is smaller than this threshold (e.g., `0.05` = skip if change < 5%). Avoids unnecessary remeshing and re-solves for negligible domain adjustments. |

### S-curve iteration control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_max` | int | `5000` | Hard upper limit on the total number of solver calls. The run stops here regardless of convergence. For typical mechanisms and default step sizes, a full S-curve usually takes a few hundred iterations; this limit is a safety net against infinite loops. |
| `n_extinction_points` | int | `10` | After the ignited (upper + unstable) branches are complete, the code traces the cold mixing (lower, extinction) branch between chi_st_min (last unstable point) and chi_st_max (10× the ignited-branch peak) using this many logarithmically spaced chi_st values. Set to `0` to skip the extinction branch entirely. |

### Two-point control: control point placement

At each iteration the solver places two temperature control points in the flame profile, lowers their target temperatures by `delta_T`, and solves. The placement algorithm determines *where* in the profile the control points sit.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loc_algo_left` | string | `"spacing"` | Algorithm for the left (fuel-side) control point. `"spacing"`: pin at `T_min + initial_spacing × (T_max − T_min)` — a fixed fractional position in the temperature range. `"max_dTdx"`: pin at the left peak of `dT/dx` — i.e., the steeper flank of the reaction zone. `"next_to_max"`: pin at the grid point immediately left of peak T — used deep on the unstable branch as the flame collapses. |
| `loc_algo_right` | string | `"spacing"` | Same options as `loc_algo_left` but for the right (oxidizer-side) control point. **Note:** after the turning point is detected, the code automatically overrides this to `"next_to_max"` regardless of user input, because the spacing algorithm becomes ill-conditioned as the flame narrows. |
| `initial_spacing` | float | `0.6` | Fractional position used by the `"spacing"` algorithm. `0.6` places both control points at 60% of the way from `T_min` to `T_max`. Higher values push the control points closer to the peak and result in tighter control of the near-peak region; lower values place them further into the preheat/post-flame zone. |

### Two-point control: step size and adaptation

`delta_T` is the temperature drop imposed on the control points each iteration. The solver adapts it automatically after each step so that the *actual* change in peak temperature stays near `target_delta_T_max`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta_T_type` | string | `"absolute"` | How `delta_T` is interpreted. `"absolute"`: control point temperatures are decremented by `delta_T` [K] — appropriate on the upper branch where large K-steps are efficient. `"relative"`: control point temperatures are decremented by `delta_T × T_control` — appropriate near extinction where the remaining temperature range is small and absolute steps would overshoot. **Note:** after the turning point, the code automatically switches to `"relative"` internally. |
| `delta_T` | float | `20.0` | Starting value of the control point temperature decrement [K for `absolute`, dimensionless fraction for `relative`]. After each successful solve this value is adapted up or down (see `target_delta_T_max`). After each failed solve it is cut by 30% (`delta_T *= 0.7`) before retrying. |
| `max_delta_T` | float | `100.0` | Upper bound on `delta_T` during adaptation. Prevents the step from growing so large that a single iteration jumps over an important region of the S-curve. |
| `target_delta_T_max` | float | `20.0` | The peak temperature change that the adaptive scheme aims for per step. After a successful solve: if `|ΔT_max| < 0.8 × target`, `delta_T` is increased by 3; if `|ΔT_max| > target`, `delta_T` is scaled down proportionally. This keeps coverage of the S-curve roughly uniform without wasting iterations on trivially small steps. |

### Convergence and termination

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_error_count` | int | `3` | Maximum *consecutive* Cantera solver failures allowed before the run aborts. On each failure the backup flame state is restored and `delta_T` is cut by 30% before retrying. If `max_error_count` failures occur in a row without a successful solve the run exits — reporting success if `strain_rate_tol` was met, or failure otherwise. |
| `strain_rate_tol` | float | `0.10` | Termination criterion for the unstable branch. The run is declared successful when `strain_rate_max / strain_rate_peak < strain_rate_tol`, i.e., when the flame has weakened to this fraction of its peak strain rate. `0.10` means the solver continues until the strain rate has dropped to 10% of its maximum — ensuring substantial coverage of the unstable branch. Increase this (e.g., `0.3`) for a shorter run that stops earlier on the unstable branch. |

---

## `[tabulation]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dims` | list[int, int, int] | `[100, 2, 100]` | Grid dimensions: `[Z (mixture fraction), Q (variance), L (progress variable)]` |
| `force_monotonicity` | boolean | `false` | Enforce strict monotonicity along the progress variable coordinate |
| `igniting_table` | boolean | `false` | Include the ignition region (affects `SRC_PROG` at low progress variable values) |
| `include_species_mass_fractions` | string or list | `[]` | Species mass fractions to include in the table: `"all"` or a list of species names |
| `include_species_production_rates` | string or list | `[]` | Species production rates to include: `"all"` or a list of species names |
| `include_energy_enthalpy_components` | boolean | `false` | Include energy/enthalpy components: `E_CHEM`, `E0_SENS`, `H0`, `H0_SENS` |
| `n_workers` | int | `null` | Number of parallel worker processes for flamelet post-processing. `null` uses all available CPU cores. Set to `1` to force serial execution (useful for debugging). Each worker loads its own Cantera gas object, so memory use scales with this value. |

---

## `[plotting.s_curve]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_quantity` | string | `"chi_st"` | X-axis quantity: `"chi_st"` or `"strain_rate_max"` |
| `y_quantity` | string | `"T_max"` | Y-axis quantity: `"T_max"` or `"T_st"` |

---

## `[plotting.profiles]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_profiles` | int | `null` | Number of flamelet profiles to plot; `null` plots all |
| `colormap` | string | `"viridis"` | Matplotlib colormap name for profile line coloring |

---

## `[plotting.table]` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vars` | list or `"all"` | `[]` | Table variables to plot; `"all"` plots every variable |
| `Q_plot` | float | `0.0` | Mixture fraction variance slice value used for 2D table plots (0.0–1.0) |
| `colormap` | string | `"viridis"` | Matplotlib colormap name for table plots |

---

## `[restart]` Section *(Optional)*

### Background

FPVgen traces the S-curve one point at a time. Every time the Cantera solver converges on a new solution, that point is immediately written to `solutions.h5` in the `output_dir`. If a run is killed or crashes partway through, all solutions up to that point are already on disk. The `[restart]` section lets you resume from that file rather than starting over.

### What `solutions.h5` contains

- **File header**: all physical setup parameters (mechanism file, transport model, fuel/oxidizer composition and temperature, pressure, width settings). These are written once when the first solution is saved.
- **Per solution (indexed 0, 1, 2, …)**: the full Cantera flame state (temperature and species profiles at every grid point), plus the mixture fraction and scalar dissipation profiles, plus metadata (T_max, chi_st, strain rates, branch_id, CPU time, etc.).

### How restart works

When FPVgen sees a `[restart]` section it:

1. Reads `solutions_file` and fully reconstructs the `FlameletTableGenerator` — including all previously computed solutions — from the file. The physical setup (mechanism, inlets, pressure) is taken from the file, **not** from the current config's `[mechanism]`/`[fuel_inlet]`/`[oxidizer_inlet]`/`[conditions]` sections.
2. If the current `output_dir` differs from the directory containing `solutions_file`, all existing solutions are copied to the new output directory before computation begins.
3. Continues computing new solutions, which are appended on top of the existing ones in both memory and in `solutions.h5`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solutions_file` | string | **Required** | Path to the `solutions.h5` file from a previous run. |
| `solution_index` | int | — | Intended to let you restart from a specific earlier solution index rather than continuing from the last one — useful if the run went off track near the end and you want to back up a few steps. **Currently documented but not wired up** in `generate_table.py`; specifying it has no effect in the present code. |

### What is and is not re-read from your config on restart

| Source | On restart |
|--------|-----------|
| Mechanism file, transport model | Read from `solutions.h5`, config ignored |
| Fuel/oxidizer composition, temperature | Read from `solutions.h5`, config ignored |
| Pressure | Read from `solutions.h5`, config ignored |
| `output_dir` | Read from `[solver]` in your config |
| All `[solver]` tracing parameters (`delta_T`, `n_max`, etc.) | Read from `[solver]` in your config — you can change these on restart |
| `[tabulation]`, `[plotting]` | Read from your config as normal |

---

## Example `input.toml`

```toml
[mechanism]
file = "gri30.yaml"
transport_model = "mixture-averaged"
prog_def = { CO = 1.0, H2 = 1.0, CO2 = 1.0, H2O = 1.0 }

[conditions]
pressure = 101325.0
initial_chi_st = 1.0e-3

[fuel_inlet]
composition = { CH4 = 1.0 }
temperature = 300.0

[oxidizer_inlet]
composition = { O2 = 0.21, N2 = 0.79 }
temperature = 300.0

[solver]
output_dir = "flamelet_results"
n_max = 5000
n_extinction_points = 10
create_plots = true

[tabulation]
dims = [100, 2, 100]
igniting_table = false
include_species_mass_fractions = []

[plotting.s_curve]
x_quantity = "chi_st"
y_quantity = "T_max"

[plotting.profiles]
num_profiles = 10
colormap = "viridis"

[plotting.table]
vars = "all"
Q_plot = 0.0
```

---

*Source: `fpvgen/flamelet_table_generator.py`, `scripts/generate_table.py`*
