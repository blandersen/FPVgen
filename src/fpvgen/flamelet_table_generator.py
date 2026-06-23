import logging
import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple, Literal
import h5py
import json
import numpy as np
import cantera as ct
from scipy import interpolate
from scipy.special import erfcinv
from scipy.signal import find_peaks

from fpvgen import coordinate

def _process_flamelet_worker(args):
    """Process a single flamelet solution for FPV table assembly.

    Runs in a subprocess; all Cantera objects are created locally so there is no
    shared mutable state.  Returns a dict mapping variable name → interpolated
    1-D array on Z_grid.
    """
    (flame_arrays, Z_i, Z_grid, mechanism_file, prog_def,
     include_species_mass_fractions, include_species_production_rates,
     include_energy_enthalpy_components) = args

    gas = ct.Solution(mechanism_file)

    def build_interp(Z, data):
        return interpolate.interp1d(
            Z, data, axis=0, bounds_error=False, fill_value=(data[0], data[-1])
        )

    T       = flame_arrays["T"]
    Y       = flame_arrays["Y"]
    P       = flame_arrays["P"]
    density = flame_arrays["density"]
    n_pts   = len(T)

    E0_i  = flame_arrays["int_energy_mass"]
    ROM_i = ct.gas_constant / flame_arrays["mean_molecular_weight"]

    results = {}
    results["ZBilger"]     = Z_grid
    results["ROM"]         = build_interp(Z_i, ROM_i)(Z_grid)
    results["T0"]          = build_interp(Z_i, T)(Z_grid)
    results["rho0"]        = build_interp(Z_i, density)(Z_grid)
    results["E0"]          = build_interp(Z_i, E0_i)(Z_grid)

    # Per-point energy perturbation to compute thermodynamic derivatives
    rho0deltaE = 5000.0
    deltaE  = rho0deltaE / density
    E0_p    = E0_i + deltaE
    E0_m    = E0_i - deltaE
    T0_p    = np.empty(n_pts)
    T0_m    = np.empty(n_pts)
    MU0_p   = np.empty(n_pts)
    MU0_m   = np.empty(n_pts)
    LOC0_p  = np.empty(n_pts)
    LOC0_m  = np.empty(n_pts)
    SRC_PROG_p = np.zeros(n_pts)
    SRC_PROG_m = np.zeros(n_pts)

    for j in range(n_pts):
        gas.TPY = T[j], P, Y[:, j]
        gas.UV  = E0_p[j], gas.v
        T0_p[j]   = gas.T
        MU0_p[j]  = gas.viscosity
        LOC0_p[j] = gas.thermal_conductivity / gas.cp_mass
        for species, value in prog_def.items():
            idx = gas.species_index(species)
            SRC_PROG_p[j] += value * gas.net_production_rates[idx] * gas.molecular_weights[idx]
        SRC_PROG_p[j] /= gas.density

        gas.TPY = T[j], P, Y[:, j]
        gas.UV  = E0_m[j], gas.v
        T0_m[j]   = gas.T
        MU0_m[j]  = gas.viscosity
        LOC0_m[j] = gas.thermal_conductivity / gas.cp_mass
        for species, value in prog_def.items():
            idx = gas.species_index(species)
            SRC_PROG_m[j] += value * gas.net_production_rates[idx] * gas.molecular_weights[idx]
        SRC_PROG_m[j] /= gas.density

    dTm   = T - T0_m
    dTp   = T0_p - T
    dT    = T0_p - T0_m
    dedT  = (  dTm**2           * (E0_i + deltaE)
             + dT  * (dTp - dTm) * E0_i
             - dTp**2           * (E0_i - deltaE)) / (dTp * dTm * dT)
    d2edT2 = 2.0 * (  dTm * (E0_i + deltaE)
                     - dT  *  E0_i
                     + dTp * (E0_i - deltaE)) / (dTp * dTm * dT)

    GAMMA0_i = ROM_i / dedT + 1.0
    results["GAMMA0"] = build_interp(Z_i, GAMMA0_i)(Z_grid)
    results["AGAMMA"] = build_interp(Z_i, -d2edT2 * (GAMMA0_i - 1.0)**2 / ROM_i)(Z_grid)

    MU0_i  = flame_arrays["viscosity"]
    LOC0_i = flame_arrays["thermal_conductivity"] / flame_arrays["cp_mass"]
    results["MU0"]  = build_interp(Z_i, MU0_i)(Z_grid)
    results["AMU"]  = build_interp(Z_i, np.log(MU0_p / MU0_m) / np.log(T0_p / T0_m))(Z_grid)
    results["LOC0"] = build_interp(Z_i, LOC0_i)(Z_grid)
    results["ALOC"] = build_interp(Z_i, np.log(LOC0_p / LOC0_m) / np.log(T0_p / T0_m))(Z_grid)

    net_prod = flame_arrays["net_production_rates"]

    # SRC_PROG and PROG use the full flame net_production_rates field (not perturbed gas)
    prog_var_prod = np.zeros(n_pts)
    prog_var      = np.zeros(n_pts)
    for species, value in prog_def.items():
        idx = gas.species_index(species)
        prog_var_prod += value * net_prod[idx, :] * gas.molecular_weights[idx]
        prog_var      += value * Y[idx, :]
    results["SRC_PROG"]   = build_interp(Z_i, prog_var_prod / density)(Z_grid)
    results["PROG"]       = build_interp(Z_i, prog_var)(Z_grid)
    results["HeatRelease"] = build_interp(
        Z_i, flame_arrays["heat_release_rate"] / density
    )(Z_grid)

    for sp in include_species_mass_fractions:
        idx = gas.species_index(sp)
        results[sp] = build_interp(Z_i, Y[idx, :])(Z_grid)

    for sp in include_species_production_rates:
        k = gas.species_index(sp)
        results["SRC_" + sp] = build_interp(
            Z_i, net_prod[k, :] * gas.molecular_weights[k] / density
        )(Z_grid)

    if include_energy_enthalpy_components:
        E_CHEM_i = np.empty(n_pts)
        for j in range(n_pts):
            gas.TPY = 298.15, P, Y[:, j]
            E_CHEM_i[j] = (
                np.dot(gas.standard_enthalpies_RT, gas.X)
                * ct.gas_constant * gas.T / gas.mean_molecular_weight
            )
        results["E_CHEM"]   = build_interp(Z_i, E_CHEM_i)(Z_grid)
        results["E0_SENS"]  = build_interp(Z_i, E0_i - E_CHEM_i)(Z_grid)
        results["H0"]       = build_interp(Z_i, flame_arrays["enthalpy_mass"])(Z_grid)
        results["H0_SENS"]  = build_interp(Z_i, flame_arrays["enthalpy_mass"] - E_CHEM_i)(Z_grid)

    TA_i = np.log(SRC_PROG_p / SRC_PROG_m) / ((1.0 / T0_p) - (1.0 / T0_m))
    TA_i = np.maximum(TA_i, 0.0)
    TA_i[np.isnan(TA_i)] = 0.0
    results["TA"] = build_interp(Z_i, TA_i)(Z_grid)

    return results


pyplot_params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
}


@dataclass
class InletCondition:
    """Represents inlet conditions for fuel or oxidizer stream"""

    composition: Dict[str, float]  # Species mole fractions
    temperature: float  # Temperature in Kelvin


class FlameletTableGenerator:
    """Generator for flamelet calculations including unstable branch traversal.

    This class handles the generation of flamelet solutions for counterflow diffusion flames,
    including the computation of complete S-curves with stable and unstable branches.

    Attributes:
        mechanism_file (str): Path to the chemical mechanism file
        transport_model (str): Transport model to use in Cantera
        fuel_inlet (InletCondition): Fuel stream inlet conditions
        oxidizer_inlet (InletCondition): Oxidizer stream inlet conditions
        pressure (float): Operating pressure in Pa
        prog_def (Dict): Progress variable definition
        width_ratio (float): Ratio of the domain width to the flame thickness
        width_change_enable (bool): Enable domain width changes
        width_change_max (float): Maximum domain width change
        width_change_min (float): Minimum domain width change
        initial_chi_st (float): Initial scalar dissipation rate at stoichiometric mixture fraction
        gas (ct.Solution): Cantera Solution object for the mechanism
        flame (ct.CounterflowDiffusionFlame): Cantera flame object
        solutions (List[Dict]): List of computed solutions and their metadata
        Z_st (float): Stoichiometric mixture fraction
        solver_loglevel (int): Cantera solver log level (0-3)
    """

    def __init__(
        self,
        mechanism_file: str,
        transport_model: str,
        fuel_inlet: InletCondition,
        oxidizer_inlet: InletCondition,
        pressure: float,
        prog_def: Optional[Dict] = None,
        width_ratio: Optional[float] = 10.0,
        width_change_enable: Optional[bool] = False,
        width_change_max: Optional[float] = 0.2,
        width_change_min: Optional[float] = 0.05,
        initial_chi_st: Optional[float] = 1.0e-3,
        solver_loglevel: Optional[int] = 0,
        strain_chi_st_model_param_file: Optional[str] = None,
    ):
        """Initialize the flamelet generator with mechanism and conditions.

        Args:
            mechanism_file: Path to the chemical mechanism file
            transport_model: Transport model to use in Cantera
            fuel_inlet: Fuel stream inlet conditions
            oxidizer_inlet: Oxidizer stream inlet conditions
            pressure: Operating pressure in Pa
            prog_def: Progress variable definition
            width_ratio: Ratio of the domain width to the flame thickness
            width_change_enable: Enable domain width changes
            width_change_max: Maximum domain width change
            width_change_min: Minimum domain width change
            initial_chi_st: Initial scalar dissipation rate at stoichiometric mixture fraction
            solver_loglevel: Cantera solver log level (0-3)
            strain_chi_st_model_param_file: Path to JSON file with strain vs chi_st model parameters
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Store input parameters
        self.mechanism_file = mechanism_file
        self.transport_model = transport_model
        self.fuel_inlet = fuel_inlet
        self.oxidizer_inlet = oxidizer_inlet
        self.pressure = pressure
        self.width_ratio = width_ratio
        self.width_change_enable = width_change_enable
        self.width_change_max = width_change_max
        self.width_change_min = width_change_min
        self.initial_chi_st = initial_chi_st
        self.solver_loglevel = solver_loglevel

        if prog_def is None:
            self.prog_def = {"CO": 1.0, "H2": 1.0, "CO2": 1.0, "H2O": 1.0}
        else:
            self.prog_def = prog_def

        if strain_chi_st_model_param_file is not None:
            with open(strain_chi_st_model_param_file, "r") as f:
                self.strain_chi_st_model_params = json.load(f)
        else:
            self.strain_chi_st_model_params = None

        # Initialization
        self.gas = ct.Solution(self.mechanism_file)
        self.gas.transport_model = self.transport_model
        self.Z_st = self._compute_stoichiometric_mixture_fraction()
        self.logger.info("Z_st = {:.8f}".format(self.Z_st))
        self.width = 1.0  # Needed for initial flame construction but will be overridden
        self.flame = None
        self.solutions = []
        self._update_flame_width()

    def _compute_stoichiometric_mixture_fraction(self) -> float:
        """Compute the stoichiometric mixture fraction using Bilger's definition.

        Returns:
            float: Stoichiometric mixture fraction
        """
        self.gas.set_equivalence_ratio(1.0, self.fuel_inlet.composition, self.oxidizer_inlet.composition)
        return self.gas.mixture_fraction(
            fuel=self.fuel_inlet.composition,
            oxidizer=self.oxidizer_inlet.composition,
            basis="mole",
            element="Bilger",
        )

    def _compute_progress_variable(self) -> np.ndarray:
        """Compute the progress variable field for the current flame solution.

        Returns:
            np.ndarray: Progress variable field
        """
        prog_var = np.zeros_like(self.flame.grid)
        for species, value in self.prog_def.items():
            idx = self.gas.species_index(species)
            prog_var += value * self.flame.Y[idx, :]
        return prog_var

    def _compute_progress_variable_production(self) -> np.ndarray:
        """Compute the progress variable production rate [kg/m^3s] field for the current flame solution.

        Returns:
            np.ndarray: Progress variable production rate field
        """
        prog_var_prod = np.zeros_like(self.flame.grid)
        for species, value in self.prog_def.items():
            idx = self.gas.species_index(species)
            prog_var_prod += value * self.flame.net_production_rates[idx, :] * self.gas.molecular_weights[idx]
        return prog_var_prod

    def _compute_scalar_dissipation(self, mixture_fraction: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Compute scalar dissipation rate field and its stoichiometric value.

        Args:
            mixture_fraction: Optional pre-computed mixture fraction field.
                If None, will be computed.

        Returns:
            Tuple containing:
                - np.ndarray: Scalar dissipation rate at each grid point
                - float: Scalar dissipation rate at stoichiometric mixture fraction
        """
        if mixture_fraction is None:
            mixture_fraction = self.flame.mixture_fraction("Bilger")

        # Compute mixture fraction gradient
        dZ_dx = np.gradient(mixture_fraction, self.flame.grid)

        # Compute diffusivity (assuming Lewis number = 1)
        rho = self.flame.density
        D = self.flame.thermal_conductivity / (rho * self.flame.cp_mass)

        # Compute scalar dissipation rate
        chi = 2 * D * dZ_dx**2

        # Find scalar dissipation at stoichiometric mixture fraction
        interp = interpolate.interp1d(mixture_fraction, chi)
        chi_st = float(interp(self.Z_st))

        return chi, chi_st

    def _compute_flame_loc_Z(self) -> float:
        """Compute the flame location in mixture fraction space.

        Returns:
            float: Mixture fraction at the flame location
        """
        Z = self.flame.mixture_fraction("Bilger")
        T = self.flame.T
        return Z[np.argmax(T)]

    def _compute_dLnCpDZ(self) -> np.ndarray:
        """Compute the derivative of the natural logarithm of the heat capacity with respect to Z.

        Returns:
            np.ndarray: Derivative of ln(Cp) with respect to Z
        """
        Z = self.flame.mixture_fraction("Bilger")[::-1]
        cp = self.flame.cp_mass[::-1]
        return np.gradient(np.log(cp), Z)[::-1]

    def _estimate_chi_st_from_strain(self, strain_rate: float) -> float:
        """Estimate the stoichiometric scalar dissipation rate from a target strain rate.

        Args:
            strain_rate: Target strain rate [1/s]

        Returns:
            float: Estimated stoichiometric scalar dissipation rate [1/s]

        Note:
            Uses regression model if available, otherwise uses Peters' relationship
            (Peters, PECS 1984)
        """
        if self.strain_chi_st_model_params is not None:
            m = self.strain_chi_st_model_params["slope"]
            b = self.strain_chi_st_model_params["intercept"]
            log_chi_st = m * np.log10(strain_rate) + b
            chi_st = 10**log_chi_st
        else:
            chi_st = (strain_rate / np.pi) * np.exp(-2 * erfcinv(2 * self.Z_st) ** 2)
        return chi_st

    def _estimate_strain_from_chi_st(self, chi_st: float) -> float:
        """Estimate the strain rate from a given stoichiometric scalar dissipation rate.

        Args:
            chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]

        Returns:
            float: Estimated strain rate [1/s]

        Note:
            Uses regression model if available, otherwise uses Peters' relationship
            (Peters, PECS 1984)
        """
        if self.strain_chi_st_model_params is not None:
            m = self.strain_chi_st_model_params["slope"]
            b = self.strain_chi_st_model_params["intercept"]
            log_strain_rate = (np.log10(chi_st) - b) / m
            strain_rate = 10**log_strain_rate
        else:
            strain_rate = chi_st * np.pi / np.exp(-2 * erfcinv(2 * self.Z_st) ** 2)
        return strain_rate

    def _estimate_flame_thickness(self, strain_rate: Optional[float] = None, chi_st: Optional[float] = None) -> float:
        """Estimate the flame thickness based on strain rate or chi_st.

        Args:
            strain_rate: Optional strain rate [1/s]
            chi_st: Optional scalar dissipation rate at stoichiometric mixture fraction [1/s]

        Returns:
            float: Estimated flame thickness [m]
        """
        if strain_rate is not None:
            chi_st = self._estimate_chi_st_from_strain(strain_rate)
            return self._estimate_flame_thickness(chi_st=chi_st)
        elif chi_st is not None:
            self.gas.TPX = (
                0.5 * (self.fuel_inlet.temperature + self.oxidizer_inlet.temperature),
                self.pressure,
                self.fuel_inlet.composition,
            )
            thermal_diffusivity = self.gas.thermal_conductivity / (self.gas.density * self.gas.cp_mass)
            return np.sqrt(thermal_diffusivity / chi_st)
        else:
            raise ValueError("Either strain_rate or chi_st must be provided.")

    def _measure_flame_thickness(self) -> float:
        """Measure the flame thickness in the current state.

        Returns:
            float: Measured flame thickness [m]
        """
        if self.flame.extinct():
            # Flame is extinguished
            return 0.0
            # # Placeholder to prevent mesh update
            # return self.width / self.width_ratio
            # # Alternative - use estimate
            # strain_rate = self.flame.strain_rate('max')
            # return self._estimate_flame_thickness(strain_rate=strain_rate)
        else:
            T_max = np.max(self.flame.T)
            T_threshold = 0.5 * T_max
            indices = np.where(self.flame.T >= T_threshold)[0]
            flame_thickness = self.flame.grid[indices[-1]] - self.flame.grid[indices[0]]
        return flame_thickness

    def _mdots_from_chi_st(self, chi_st: float) -> Tuple[float, float]:
        """Compute mass fluxes for fuel and oxidizer based on target chi_st.

        Args:
            chi_st: Target scalar dissipation rate at stoichiometric mixture fraction [1/s]

        Returns:
            Tuple containing:
                - mdot_fuel: Mass flux of fuel [kg/m²/s]
                - mdot_oxidizer: Mass flux of oxidizer [kg/m²/s]
        """
        # Set gas state for fuel
        self.gas.TPX = (self.fuel_inlet.temperature, self.pressure, self.fuel_inlet.composition)
        rho_fuel = self.gas.density

        # Set gas state for oxidizer
        self.gas.TPX = (
            self.oxidizer_inlet.temperature,
            self.pressure,
            self.oxidizer_inlet.composition,
        )
        rho_ox = self.gas.density

        # Estimate strain rate needed for target chi_st
        target_strain = self._estimate_strain_from_chi_st(chi_st)

        # Set velocities to achieve target strain while maintaining momentum balance
        v_tot = target_strain * self.width / 2
        v_fuel = v_tot / (1 + np.sqrt(rho_fuel / rho_ox))
        v_ox = v_tot - v_fuel

        # Convert to mass fluxes
        mdot_fuel = rho_fuel * v_fuel
        mdot_oxidizer = rho_ox * v_ox

        return mdot_fuel, mdot_oxidizer

    def _strain_rate_nominal(self) -> float:
        """Compute the nominal strain rate based on the input velocities.

        Returns:
            float: Nominal strain rate [1/s]
        """
        self.gas.TPX = (self.fuel_inlet.temperature, self.pressure, self.fuel_inlet.composition)
        rho_fuel = self.gas.density
        self.gas.TPX = (
            self.oxidizer_inlet.temperature,
            self.pressure,
            self.oxidizer_inlet.composition,
        )
        rho_ox = self.gas.density
        v_fuel = self.flame.fuel_inlet.mdot / rho_fuel
        v_ox = self.flame.oxidizer_inlet.mdot / rho_ox
        return 2 * (v_fuel + v_ox) / self.width

    def _initialize_flame(
        self,
        chi_st: float,
        grid: Optional[np.ndarray] = None
    ) -> None:
        """Set up the initial counterflow diffusion flame configuration.

        Args:
            chi_st: The scalar dissipation rate at the stoichiometric mixture fraction
            grid: The grid for the flame object

        Initializes the Cantera flame object with appropriate grid, inlet conditions,
        and refinement criteria. Estimates appropriate strain rate based on target
        scalar dissipation rate.
        """
        if grid is not None:
            self.flame = ct.CounterflowDiffusionFlame(self.gas, grid=grid)
        else:
            self.flame = ct.CounterflowDiffusionFlame(self.gas, width=self.width)
        self.flame.transport_model = self.transport_model

        # Set operating conditions
        self.flame.P = self.pressure

        # Set inlet conditions
        mdot_fuel, mdot_ox = self._mdots_from_chi_st(chi_st)
        self.flame.fuel_inlet.mdot = mdot_fuel
        self.flame.fuel_inlet.X = self.fuel_inlet.composition
        self.flame.fuel_inlet.T = self.fuel_inlet.temperature
        self.flame.oxidizer_inlet.mdot = mdot_ox
        self.flame.oxidizer_inlet.X = self.oxidizer_inlet.composition
        self.flame.oxidizer_inlet.T = self.oxidizer_inlet.temperature

        # Set refinement parameters
        self.flame.set_refine_criteria(ratio=4.0, slope=0.1, curve=0.2, prune=0.05)

    def _enable_two_point_control(self) -> None:
        self.flame.two_point_control_enabled = True
        self.flame.flame.set_bounds(spread_rate=(-1e-5, 1e20))
        self.flame.max_time_step_count = 100

    def _update_flame_width(self, solve: Optional[bool] = True) -> None:
        """Update the flame width and reinitialize the flame object.

        Args:
            solve: Whether to solve to steady state after update
        """
        if self.flame is None:
            flame_thickness = self._estimate_flame_thickness(chi_st=self.initial_chi_st)
        else:
            flame_thickness = self._measure_flame_thickness()

        # Compute the new width
        old_width = self.width
        target_width = self.width_ratio * flame_thickness
        if self.flame is None:
            self.width = target_width
            self._initialize_flame(self.initial_chi_st)
            return
        if not self.width_change_enable:
            return
        if np.abs(target_width - old_width) / old_width <= self.width_change_min:
            return
        self.width = np.clip(
            target_width,
            (1 + self.width_change_max) * old_width,
            1 / (1 + self.width_change_max) * old_width,
        )
        if self.width == old_width:
            return

        width_increasing = self.width >= old_width
        self.logger.info(f"Updating domain width from {old_width:.3e} m to {self.width:.3e} m")

        # Save current state
        old_solution = self.flame.to_array()
        old_mdots = (self.flame.fuel_inlet.mdot, self.flame.oxidizer_inlet.mdot)
        old_grid = self.flame.grid

        # Find approximate flame location (using peak temperature)
        old_grid_norm = old_grid / old_width
        flame_idx = np.argmax(self.flame.T)
        flame_loc_old = old_grid[flame_idx]
        flame_loc_normalized = old_grid_norm[flame_idx]
        flame_loc_new = flame_loc_normalized * self.width

        # Construct new grid maintaining resolution and flame position
        if width_increasing:
            # For width increase, extend the existing grid
            # Create new grid, keeping flame_loc_normalized and absolute old spacing
            new_grid = old_grid + (flame_loc_new - flame_loc_old)

            # Fill in the gaps at the sides
            dx_l = old_grid[1] - old_grid[0]
            dx_r = old_grid[-1] - old_grid[-2]
            grid_l = np.arange(0, new_grid[0], dx_l)
            grid_r = np.arange(new_grid[-1], self.width, dx_r)
            if len(grid_r) > 0 and grid_r[-1] < self.width:
                grid_r = np.append(grid_r[1:], self.width)
            new_grid = np.concatenate((grid_l, new_grid, grid_r))
        else:
            # For width decrease, trim existing grid
            # Create new grid, keeping flame_loc_normalized and absolute old spacing
            new_grid = old_grid + (flame_loc_new - flame_loc_old)

            # Find points that fall within new domain
            interior_mask = (new_grid > 0) & (new_grid < self.width)
            new_grid = new_grid[interior_mask]

            # Add boundary points
            new_grid = np.concatenate(([0.0], new_grid, [self.width]))

        new_grid_norm = new_grid / self.width
        self._initialize_flame(chi_st=self.initial_chi_st, grid=new_grid)
        # ^ uses initial_chi_st but mdots will be overwritten below

        # Interpolate solution onto new grid
        scale_factor = self.width / old_width

        var_names = ["velocity", "spread_rate", "lambda", "T"]
        var_names += self.gas.species_names
        if self.flame.two_point_control_enabled:
            var_names += ["Uo"]

        for var_name in var_names:
            old_values = getattr(old_solution, var_name)
            interp = interpolate.interp1d(
                old_grid_norm,
                old_values,
                kind="cubic",
                bounds_error=False,
                fill_value=(old_values[0], old_values[-1]),
            )
            new_values = interp(new_grid_norm)

            # Handle special cases
            if var_name in ["velocity", "spread_rate", "Uo"]:
                new_values *= scale_factor
            # elif var_name in self.gas.species_names:
            #     new_values = np.clip(new_values, 0, 1)

            # Update the flame solution
            self.flame.set_profile(var_name, new_grid_norm, new_values)

        # # Normalize mass fractions
        # Y_sum = np.zeros_like(new_grid)
        # for k in range(self.gas.n_species):
        #     Y_sum += self.flame.Y[k, :]
        # for k in range(self.gas.n_species):
        #     Y = self.flame.Y[k, :]
        #     Y_normalized = np.where(Y_sum > 0, Y / Y_sum, 0)
        #     self.flame.set_profile(self.gas.species_names[k],
        #                            new_grid_norm,
        #                            Y_normalized)

        # Update inlet mass flow rates
        self.flame.fuel_inlet.mdot = old_mdots[0] * scale_factor
        self.flame.oxidizer_inlet.mdot = old_mdots[1] * scale_factor

        # Solve to steady state
        if solve:
            self.logger.info("Computing the solution in the new domain")
            self.flame.solve(loglevel=self.solver_loglevel, auto=True)
    
    def _update_control_points(
        self,
        loc_algo_left: str = "spacing",
        loc_algo_right: str = "spacing",
        spacing: float = 0.6,
        delta_T_type: str = "absolute",
        delta_T: float = 20.0,
    ) -> None:
        """Update the left and right control points for temperature control.
        
        Computes the left and right control temperatures based on the specified
        location algorithms and sets them in the flame object. The control points
        are adjusted based on the specified delta_T type (absolute or relative).

        Args:
            loc_algo_left: Algorithm to determine left control point location
                Options: "spacing", "max_dTdx", "next_to_max"
            loc_algo_right: Algorithm to determine right control point location
                Options: "spacing", "max_dTdx", "next_to_max"
            spacing: Fraction of the maximum temperature to set control points
            delta_T_type: Type of delta_T adjustment ("absolute" or "relative")
            delta_T: Temperature change applied to control points
        """
        # Update control temperatures
        if loc_algo_left == "spacing":
            # Sets it such that the left and right are at a temperature of spacing * max(T) 
            control_temperature_left = np.min(self.flame.T) + spacing * (np.max(self.flame.T) - np.min(self.flame.T))

        elif loc_algo_left == "max_dTdx":
            # Find the left and right points with the maximum dT/dx and set them as the locations to apply temperature control
            dTdx = np.diff(self.flame.T)/np.diff(self.flame.grid)
            peaks, _ = find_peaks(dTdx) # find the index of the peaks
            peak_values = dTdx[peaks] # find the values of the peaks
            top2_idx_in_peaks = np.argsort(peak_values)[-2:] # find the largest 2 peaks in the array and return their index
            top2_peaks = peaks[top2_idx_in_peaks] # find the index of the largest 2 peaks in the original array
            top2_peaks_sorted = np.sort(top2_peaks) # sort the indices of the largest 2 peaks to ensure that the [0] index is the left peak and [1] index is the right peak
            control_temperature_left = self.flame.T[top2_peaks_sorted[0]]
        
        elif loc_algo_left == "next_to_max":
            # Sets it such that the control temperature points are always the grid points next to the maximum T
            max_T_idx = np.argmax(self.flame.T)
            control_temperature_left = self.flame.T[max_T_idx - 1]
        
        if loc_algo_right == "spacing":
            # Sets it such that the left and right are at a temperature of spacing * max(T) 
            control_temperature_right  = np.min(self.flame.T) + spacing * (np.max(self.flame.T) - np.min(self.flame.T))

        elif loc_algo_right == "max_dTdx":
            # Find the left and right points with the maximum dT/dx and set them as the locations to apply temperature control
            dTdx = np.diff(self.flame.T)/np.diff(self.flame.grid)
            peaks, _ = find_peaks(dTdx) # find the index of the peaks
            peak_values = dTdx[peaks] # find the values of the peaks
            top2_idx_in_peaks = np.argsort(peak_values)[-2:] # find the largest 2 peaks in the array and return their index
            top2_peaks = peaks[top2_idx_in_peaks] # find the index of the largest 2 peaks in the original array
            top2_peaks_sorted = np.sort(top2_peaks) # sort the indices of the largest 2 peaks to ensure that the [0] index is the left peak and [1] index is the right peak
            control_temperature_right = self.flame.T[top2_peaks_sorted[1]]
        
        elif loc_algo_right == "next_to_max":
            # Sets it such that the control temperature points are always the grid points next to the maximum T
            max_T_idx = np.argmax(self.flame.T)
            control_temperature_right = self.flame.T[max_T_idx + 1]

        self.flame.set_left_control_point(control_temperature_left)
        self.flame.set_right_control_point(control_temperature_right)
        if delta_T_type == "absolute":
            self.flame.left_control_point_temperature -= delta_T
            self.flame.right_control_point_temperature -= delta_T
        elif delta_T_type == "relative":
            self.flame.left_control_point_temperature -= self.flame.left_control_point_temperature * delta_T
            self.flame.right_control_point_temperature -= self.flame.right_control_point_temperature * delta_T

    def compute_s_curve(
        self,
        output_dir: Optional[Path] = None,
        n_extinction_points: int = 10,
        write_FlameMaster: bool = False,
        **kwargs,
    ) -> List[Dict]:
        """Compute complete S-curve including ignited branches and extinction branch.

        Computes the full S-curve by first traversing the upper (stable) and middle
        (unstable) branches, then computing the lower (extinction) branch.

        Args:
            output_dir: Directory to save solution files
            n_extinction_points: Number of points to compute along extinction branch
            write_FlameMaster: Whether to write FlameMaster output files for each solution
            **kwargs: Additional arguments passed to compute_ignited_branches

        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point
        """
        # First compute the ignited and unstable branches
        self.logger.info("Computing ignited and unstable branches")
        ignited_data = self.compute_ignited_branches(
            output_dir=output_dir, write_FlameMaster=write_FlameMaster, **kwargs
        )

        # Get chi_st bounds for extinction branch
        chi_st_values = [sol["chi_st"] for sol in ignited_data]
        chi_st_max = max(chi_st_values) * 1.0e1
        chi_st_min = ignited_data[-1]["chi_st"]  # Last point on unstable branch

        # Compute extinction branch
        self.logger.info("Computing extinction branch")
        if n_extinction_points > 0:
            extinct_data = self.compute_extinct_branch(
                chi_st_min=chi_st_min,
                chi_st_max=chi_st_max,
                n_points=n_extinction_points,
                output_dir=output_dir,
                write_FlameMaster=write_FlameMaster,
            )
        else:
            extinct_data = []

        return ignited_data + extinct_data

    def compute_ignited_branches(
        self,
        output_dir: Optional[Path] = None,
        restart_from: Optional[int] = None,
        n_max: int = 5000,
        loc_algo_left: str = "spacing",
        loc_algo_right: str = "spacing",
        initial_spacing: float = 0.6,
        delta_T_type: str = "absolute",
        delta_T: float = 20.0,
        max_delta_T: float = 100.0,
        target_delta_T_max: float = 20.0,
        max_error_count: int = 3,
        strain_rate_tol: float = 0.10,
        write_FlameMaster: bool = False,
    ) -> List[Dict]:
        """Compute upper (stable) and middle (unstable) branches of the S-curve.

        This method traverses both the upper (stable) and middle (unstable) branches of the
        S-curve using temperature as a control parameter. It employs a two-point continuation
        method to track solutions through the turning point and down the unstable branch.

        Args:
            output_dir: Directory to save solution files
            n_max: Maximum number of iterations before stopping
            loc_algo_left: Algorithm to use to calculate left control point
            loc_algo_right: Algorithm to use to calculate right control point
            initial_spacing: Initial control point spacing for stable branch (0-1)
            delta_T_type: Choose whether to enforce absolute delta_T [K] or relative delta_T [fraction]
            delta_T: Initial temperature change between solutions [K]
            max_delta_T: Maximum allowed delta_T [K]
            target_delta_T_max: Target maximum temperature change per step [K]
            max_error_count: Maximum number of successive solver errors before stopping
            strain_rate_tol: Tolerance for minimum strain rate relative to maximum
            write_FlameMaster: Whether to write FlameMaster output files for each solution

        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate_max: Maximum strain rate [1/s]
                - strain_rate_nom: Nominal strain rate [1/s]
                - chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
                - total_heat_release_rate: Integrated heat release rate [W/m³]
                - n_points: Number of grid points
                - flame_width: Width of the flame [m]
                - Tc_increment: Temperature increment used for this solution [K]
                - time_steps: Number of time steps taken by solver
                - eval_count: Number of right-hand side evaluations
                - cpu_time: Total CPU time for solution [s]
                - errors: Number of solver errors encountered

        Note:
            The method uses a two-point continuation strategy where control points are placed
            at specified fractions between the minimum and maximum temperatures. The temperature
            increment is adaptively adjusted based on solution behavior and convergence.
            Solutions are saved to HDF5 files if output_dir is provided.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Handle restart case
        if restart_from is not None:
            if restart_from >= len(self.solutions):
                raise ValueError(f"Restart index {restart_from} exceeds number of available solutions")

            # Restore flame state and get previous temperature increment
            restart_solution = self.solutions[restart_from]
            self.flame.from_array(restart_solution["state"])

            # Initialize data with previous solutions
            data = [sol["metadata"] for sol in self.solutions[: restart_from + 1]]

            self.logger.info(f"Restarting from solution {restart_from}")
            start_iteration = restart_from + 1

        else:
            self.logger.info("Computing the initial solution")
            self.flame.solve(loglevel=self.solver_loglevel, auto=True)
            T_max = np.max(self.flame.T)
            Z = self.flame.mixture_fraction("Bilger")
            chi, chi_st = self._compute_scalar_dissipation(Z)
            chi_st_max_glob = chi_st
            strain_rate_max = self.flame.strain_rate("max")
            strain_rate_max_max = strain_rate_max
            strain_rate_max_min = strain_rate_max
            data = []
            start_iteration = 0

        self.logger.info("Beginning s-curve computation")
        error_count = 0
        branch_id = 1
        spacing = initial_spacing
        strain_rate_increasing = True
        for i in range(start_iteration, n_max):
            # Update flame width if we are attempting a new point
            if error_count == 0:
                self._update_flame_width(solve=True)
                self._enable_two_point_control()
                T_max = np.max(self.flame.T)

            backup_state = self.flame.to_array()

            # Update control points
            self._update_control_points(
                loc_algo_left=loc_algo_left,
                loc_algo_right=loc_algo_right,
                spacing=spacing,
                delta_T_type=delta_T_type,
                delta_T=delta_T,
            )
            self.logger.debug(f"Iteration {i}: Control temperatures = [",
                              f"{self.flame.left_control_point_temperature:.2f}, "
                              f"{self.flame.right_control_point_temperature:.2f}]")
            self.flame.clear_stats()
            T_threshold = 10.0
            if (
                self.flame.left_control_point_temperature < self.flame.fuel_inlet.T + T_threshold
                or self.flame.right_control_point_temperature < self.flame.oxidizer_inlet.T + T_threshold
            ):
                if spacing > (1 - 1e-3):
                    self.logger.info("SUCCESS! Control point temperature near inlet temperature.")
                    break
                spacing = 1 - (0.7 * (1 - spacing))

            try:
                self.flame.solve(loglevel=self.solver_loglevel)

                # Adjust temperature increment based on convergence
                if abs(max(self.flame.T) - T_max) < 0.8 * target_delta_T_max:
                    delta_T = min(delta_T + 3, max_delta_T)
                elif abs(max(self.flame.T) - T_max) > target_delta_T_max:
                    delta_T *= 0.9 * target_delta_T_max / abs(max(self.flame.T) - T_max)
                error_count = 0

            except ct.CanteraError as err:
                self.logger.debug(err)

                # Restore previous solution and reduce increment
                self.flame.from_array(backup_state)
                delta_T = 0.7 * delta_T
                error_count += 1
                if error_count > max_error_count:
                    if strain_rate_max / strain_rate_max_max < strain_rate_tol:
                        self.logger.info(
                            "SUCCESS! Traversed unstable branch down to "
                            f"{100 * strain_rate_max / strain_rate_max_max:.4f}% of the maximum strain rate."
                        )
                    else:
                        self.logger.warning(f"FAILURE! Stopping after {error_count} successive solver errors.")
                    break
                self.logger.warning(
                    f"Solver did not converge on iteration {i}. " f"Trying again with dT = {delta_T:.2f}"
                )
                continue

            # Compute postprocessing data
            Z = self.flame.mixture_fraction("Bilger")
            chi, chi_st = self._compute_scalar_dissipation(Z)
            chi_st_max_glob = max(chi_st_max_glob, chi_st)
            interp_T = interpolate.interp1d(Z, self.flame.T)
            T_st = float(interp_T(self.Z_st))
            T_max = max(self.flame.T)
            width = self._measure_flame_thickness()
            strain_rate_max = self.flame.strain_rate("max")
            strain_rate_nom = self._strain_rate_nominal()
            strain_rate_max_max = max(strain_rate_max, strain_rate_max_max)
            strain_rate_max_min = min(strain_rate_max, strain_rate_max_min)
            if len(self.solutions) > 0 and strain_rate_increasing and strain_rate_max < strain_rate_max_max:
                self.logger.info(f"Turning point encountered")
                branch_id += 1
                strain_rate_increasing = False
                loc_algo_right = "next_to_max"
                delta_T_type = "relative"
                target_delta_T_max = 0.02
                max_delta_T = 0.05
                delta_T = 0.005
            step_data = {
                "T_max": T_max,
                "T_st": T_st,
                "strain_rate_max": strain_rate_max,
                "strain_rate_nom": strain_rate_nom,
                "chi_st": chi_st,
                "total_heat_release_rate": np.trapz(self.flame.heat_release_rate, self.flame.grid),
                "n_points": len(self.flame.grid),
                "flame_width": width,
                "Tc_increment": delta_T,
                "branch_id": branch_id,
                "time_steps": sum(self.flame.time_step_stats),
                "eval_count": sum(self.flame.eval_count_stats),
                "cpu_time": sum(self.flame.jacobian_time_stats + self.flame.eval_time_stats),
                "errors": error_count,
            }
            data.append(step_data)
            self.solutions.append({"state": self.flame.to_array(), "Z": Z, "chi": chi, "metadata": step_data})

            if output_dir:
                self.save_solution(output_dir, len(self.solutions) - 1)
                if write_FlameMaster:
                    self.write_FlameMaster(output_dir)

            # Logging after successful solution
            self.logger.info(
                f"Iteration {i} completed: T_max = {T_max:.2f}, "
                f"chi_st = {chi_st:.4e}, "
                f"strain_rate_nom = {strain_rate_nom:.4e}"
            )

            if chi_st < self.solutions[0]["metadata"]["chi_st"] and not self.width_change_enable:
                self.logger.info("SUCCESS! Traversed unstable branch down to initial chi_st.")
                self.logger.info(
                    "Stopping because width changes are disabled. (Flame will start to grow beyond domain.)"
                )
                break

        self.logger.info(f"Stopped after {i} iterations")
        self.logger.info(f"Solutions computed: {len(data)}")
        self.logger.info(f"Turning points encountered: {branch_id - 1}")
        return data

    def compute_extinct_branch(
        self,
        chi_st_min: float,
        chi_st_max: float,
        n_points: int = 10,
        output_dir: Optional[Path] = None,
        write_FlameMaster: bool = False,
    ) -> List[Dict]:
        """Compute the extinction (lower) branch of the S-curve.

        This method calculates solutions along the lower (extinction) branch of the S-curve
        by starting from a cold mixing solution and gradually increasing the scalar
        dissipation rate from chi_st_min to chi_st_max. The solutions are computed at
        geometrically spaced intervals of scalar dissipation rate.

        Args:
            chi_st_min: Minimum scalar dissipation rate at stoichiometric mixture fraction [1/s]
            chi_st_max: Maximum scalar dissipation rate at stoichiometric mixture fraction [1/s]
            n_points: Number of points to compute along the extinction branch
            output_dir: Optional directory path to save solution files
            write_FlameMaster: Whether to write FlameMaster output files for each solution

        Returns:
            List[Dict]: List of solution metadata dictionaries containing properties
                of each computed solution point. Each dictionary includes:
                - T_max: Maximum temperature [K]
                - T_st: Temperature at stoichiometric mixture fraction [K]
                - strain_rate_max: Maximum strain rate [1/s]
                - strain_rate_nom: Nominal strain rate [1/s]
                - chi_st: Scalar dissipation rate at stoichiometric mixture fraction [1/s]
                - total_heat_release_rate: Integrated heat release rate [W/m³]
                - n_points: Number of grid points
                - flame_width: Width of the flame [m]
                - branch: 'extinction' to identify the branch

        Note:
            The extinction branch is computed by starting from a cold mixing solution
            and using strain rate as a control parameter to achieve target scalar
            dissipation rates. Solutions are saved to HDF5 files if output_dir
            is provided.
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        chi_st_values = np.logspace(np.log10(chi_st_min), np.log10(chi_st_max), n_points)
        chi_st_values = chi_st_values[::-1]
        data = []

        iter = 0
        while iter < 3:
            self._initialize_flame(chi_st=chi_st_max)
            try:
                self.flame.solve(loglevel=self.solver_loglevel)
            except:
                self.logger.error(f"Failed to compute initial solution at chi_st = {chi_st_max:.2e}")
                return data

            if self.flame.extinct():
                self.logger.info(f"Computed initial solution at chi_st = {chi_st_max:.2e} as cold mixing")
                break

            T_max = np.max(self.flame.T)
            self.logger.warning(
                f"Failed to compute initial solution at chi_st = {chi_st_max:.2e} (autoignited, T_max = {T_max:.2f})"
            )
            chi_st_max = 1.0e1 * chi_st_max
            chi_st_values = np.geomspace(chi_st_min, chi_st_max, n_points)
            chi_st_values = chi_st_values[::-1]
            self.logger.info(f"Retrying with chi_st_max = {chi_st_max:.2e}")

        for i, chi_st_i in enumerate(chi_st_values):
            mdot_fuel, mdot_ox = self._mdots_from_chi_st(chi_st_i)
            self.flame.fuel_inlet.mdot = mdot_fuel
            self.flame.oxidizer_inlet.mdot = mdot_ox

            try:
                self.flame.solve(loglevel=self.solver_loglevel)
            except:
                self.logger.warning(f"Failed to compute solution at chi_st = {chi_st_i:.2e}")
                continue

            if not self.flame.extinct():
                T_max = np.max(self.flame.T)
                self.logger.warning(
                    f"Failed to compute solution at chi_st = {chi_st_i:.2e} (autoignited, T_max = {T_max:.2f})"
                )
                continue

            # Compute postprocessing data
            Z = self.flame.mixture_fraction("Bilger")
            chi, chi_st = self._compute_scalar_dissipation(Z)
            interp_T = interpolate.interp1d(Z, self.flame.T)
            T_st = float(interp_T(self.Z_st))
            T_max = max(self.flame.T)
            width = self._measure_flame_thickness()
            strain_rate_max = self.flame.strain_rate("max")
            strain_rate_nom = self._strain_rate_nominal()
            step_data = {
                "T_max": T_max,
                "T_st": T_st,
                "strain_rate_max": strain_rate_max,
                "strain_rate_nom": strain_rate_nom,
                "chi_st": chi_st,
                "total_heat_release_rate": np.trapz(self.flame.heat_release_rate, self.flame.grid),
                "n_points": len(self.flame.grid),
                "flame_width": width,
                "branch_id": 0,
                "time_steps": sum(self.flame.time_step_stats),
                "eval_count": sum(self.flame.eval_count_stats),
                "cpu_time": sum(self.flame.jacobian_time_stats + self.flame.eval_time_stats),
            }
            data.append(step_data)
            self.solutions.append({
                "state": self.flame.to_array(),
                "Z": Z,
                "chi": chi,
                "metadata": step_data
            })

            if output_dir:
                self.save_solution(output_dir, len(self.solutions) - 1)
                if write_FlameMaster:
                    self.write_FlameMaster(output_dir)

            # Logging after successful solution
            self.logger.info(
                f"Iteration {i} completed: T_max = {T_max:.2f}, "
                f"chi_st = {chi_st:.4e}, "
                f"strain_rate_nom = {strain_rate_nom:.2f}"
            )

        self.logger.info(f"Completed {len(data)} points on the extinction branch")
        return data

    def save_solution(self, output_dir: Path, solution_index: int, filename: str = "solutions.h5") -> None:
        """Save a single solution to the HDF5 files.

        Saves both the flame profiles and associated metadata for a single solution.

        Args:
            output_dir: Directory path where files will be saved
            solution_index: Index of the solution being saved
            filename: Name of the HDF5 file to write within output_dir
        """
        solutions_file = output_dir / filename
        solution = self.solutions[solution_index]
        meta_name = f"meta_{solution_index:04d}"
        state_name = f"solution_state_{solution_index:04d}"

        # If this is the first solution, create the file, overwriting if necessary
        if solution_index == 0:
            # Write condition parameters
            with h5py.File(solutions_file, "w") as f:
                params = {
                    "mechanism_file": self.mechanism_file,
                    "transport_model": self.transport_model,
                    "pressure": self.pressure,
                    "width_ratio": self.width_ratio,
                    "width_change_enable": self.width_change_enable,
                    "width_change_max": self.width_change_max,
                    "width_change_min": self.width_change_min,
                    "initial_chi_st": self.initial_chi_st,
                    "solver_loglevel": self.solver_loglevel,
                    "prog_def": self.prog_def,
                    "Z_st": float(self.Z_st),
                    "fuel_inlet": {
                        "composition": self.fuel_inlet.composition,
                        "temperature": self.fuel_inlet.temperature,
                    },
                    "oxidizer_inlet": {
                        "composition": self.oxidizer_inlet.composition,
                        "temperature": self.oxidizer_inlet.temperature,
                    },
                    "strain_chi_st_model_params": self.strain_chi_st_model_params,
                }
                f.attrs["parameters"] = json.dumps(params)
                f.create_group("solutions_meta")

        # Write the solution metadata
        with h5py.File(solutions_file, "a") as f:
            # Delete existing solution group if it exists
            meta_group = f["solutions_meta"]
            if meta_name in meta_group:
                del meta_group[meta_name]

            # Create new meta group
            sol_group = meta_group.create_group(meta_name)

            # Save mixture fraction and scalar dissipation
            sol_group.create_dataset("Z", data=solution["Z"])
            sol_group.create_dataset("chi", data=solution["chi"])

            # Convert numpy values in metadata to native Python types
            metadata = {}
            for key, value in solution["metadata"].items():
                if isinstance(value, np.number):
                    metadata[key] = value.item()
                else:
                    metadata[key] = value

            # Save solution metadata
            sol_group.attrs["metadata"] = json.dumps(metadata)

        # Write flame profile
        solution["state"].save(solutions_file, name=state_name, overwrite=True)

    def save_all_solutions(self, output_dir: Path, filename: str = "solutions.h5") -> None:
        """Save all computed solutions to HDF5 files.

        Args:
            output_dir: Directory path where files will be saved
            filename: Name of the HDF5 file to write within output_dir
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(self.solutions)):
            self.save_solution(output_dir, i, filename=filename)

    def _save_solution_subset(self, output_dir: Path, indices: List[int], filename: str) -> None:
        """Save a contiguously-renumbered subset of solutions to an HDF5 file.

        The kept solutions are written under indices 0..len(indices)-1 so the resulting
        file is a valid solutions file that ``load_solutions`` can read back unchanged.
        ``self.solutions`` is not mutated.

        Args:
            output_dir: Directory path where the file will be saved
            indices: Sorted list of solution indices to keep
            filename: Name of the HDF5 file to write within output_dir
        """
        original_solutions = self.solutions
        try:
            self.solutions = [original_solutions[i] for i in indices]
            self.save_all_solutions(output_dir, filename=filename)
        finally:
            self.solutions = original_solutions

    @classmethod
    def load_solutions(
        cls,
        filename: str,
    ) -> "FlameletTableGenerator":
        """Load a complete solution set from the HDF5 file.

        Args:
            filename: Path to the solutions HDF5 file

        Returns:
            FlameletTableGenerator: New instance with loaded solutions

        Raises:
            ValueError: If files cannot be read or are invalid
        """
        # Load the metadata
        with h5py.File(filename, "r") as f:
            # Load parameters
            params = json.loads(f.attrs["parameters"])

            # Create generator instance
            generator = cls(
                mechanism_file=params["mechanism_file"],
                transport_model=params["transport_model"],
                fuel_inlet=InletCondition(**params["fuel_inlet"]),
                oxidizer_inlet=InletCondition(**params["oxidizer_inlet"]),
                pressure=params["pressure"],
                prog_def=params["prog_def"],
                width_ratio=params["width_ratio"],
                width_change_enable=params["width_change_enable"],
                width_change_max=params["width_change_max"],
                width_change_min=params["width_change_min"],
                initial_chi_st=params["initial_chi_st"],
                solver_loglevel=params["solver_loglevel"],
            )

            # Set additional attributes
            generator.Z_st = params["Z_st"]
            generator.strain_chi_st_model_params = params.get("strain_chi_st_model_params")

            # Load solutions
            generator.solutions = []
            meta_group = f["solutions_meta"]
            for meta_name in sorted(meta_group.keys()):
                sol_group = meta_group[meta_name]
                state_array = ct.SolutionArray(generator.gas)
                solution = {
                    "state": state_array,
                    "Z": sol_group["Z"][:],
                    "chi": sol_group["chi"][:],
                    "metadata": json.loads(sol_group.attrs["metadata"]),
                }
                generator.solutions.append(solution)

        # Load the states
        for solution_index, solution in enumerate(generator.solutions):
            state_name = f"solution_state_{solution_index:04d}"
            generator.solutions[solution_index]["state"].restore(filename, state_name)

        return generator

    def write_FlameMaster(
        self,
        output_dir: Path,
        solution_index: int = -1,
        reinterp_Z: bool = True,
    ) -> None:
        """Write FlameMaster output files for a single solution.

        Args:
            output_dir: Directory path where files will be saved
            solution_index: Index of the solution to write (default: last solution)
            reinterp_Z: Whether to reinterpolate the solution onto a new Z grid. This prevents issues related to dZ=0 portions of the solution.
        """
        solution = self.solutions[solution_index]

        # Build filename
        fuel_name = [key for key in self.fuel_inlet.composition.keys()][0]
        pressure_str = f"p{(self.pressure/1e5):04.1f}".replace(".", "_")
        chi_str = f"chi{solution['metadata']['chi_st']:0.4e}"
        tf_str = f"tf{self.fuel_inlet.temperature:04.0f}"
        to_str = f"to{self.oxidizer_inlet.temperature:04.0f}"
        Tst_str = f"Tst{solution['metadata']['T_st']:04.0f}"
        flamemaster_dir = output_dir / "FlameMaster"
        flamemaster_dir.mkdir(parents=True, exist_ok=True)
        filename = flamemaster_dir / f"{fuel_name}_{pressure_str}{chi_str}{tf_str}{to_str}{Tst_str}"

        Z_flame = self.flame.mixture_fraction("Bilger")

        if reinterp_Z:
            N_points = 500
            i_cut = N_points // 3
            Z_new = coordinate.CoordinateLinearThenStretched("Z", 0, self.Z_st, 1, i_cut, N_points - i_cut)
        else:
            N_points = len(Z_flame)

        def build_interp(Z, data):
            return interpolate.interp1d(Z, data, axis=0, bounds_error=False, fill_value=(data[0], data[-1]))

        def write_array(f, name, data, n_cols=5):
            if reinterp_Z:
                interp = build_interp(Z_flame, data)
                data_write = interp(Z_new.grid)
            else:
                data_write = data

            f.write(f"{name}\n")
            i_col = 0
            for i in range(len(data_write)):
                if i_col == n_cols:
                    f.write("\n")
                    i_col = 0
                f.write(f"\t{data_write[i]:0.6e}")
                i_col += 1
            f.write("\n")

        # Write the FlameMaster output file
        # fmt: off
        with open(filename, "w") as f:
            # Write header
            f.write(f"header\n")
            f.write(f"\n")
            f.write(f'title = "planar counterflow diffusion flame"\n')
            f.write(f'mechanism = "{self.mechanism_file}"\n')
            f.write(f"author = \"FPVgen\"\n")
            f.write(f"date = \"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\"\n")
            f.write(f"\n")
            f.write(f"fuel = {fuel_name}\n")
            f.write(f"pressure = {self.pressure/1e5} [bar]\n")
            f.write(f"Z_st = {self.Z_st} [1/s]\n")
            f.write(f"chi_st = {solution['metadata']['chi_st']} [1/s]\n")
            f.write(f'ConstantLewisNumbers = "False"\n')
            f.write(f"FlameLoc = {self._compute_flame_loc_Z():0.5e}\n")
            f.write(f"Tmax = {solution['metadata']['T_max']} [K]\n")
            f.write(f"\n")
            f.write(f"FuelSide\n")
            f.write(f"begin\n")
            f.write(f"\tTemperature = {self.fuel_inlet.temperature} [K]\n")
            for key, value in self.fuel_inlet.composition.items():
                f.write(f"\tMassFraction-{key} = {value}\n")
            f.write(f"end\n")
            f.write(f"\n")
            f.write(f"OxidizerSide\n")
            f.write(f"begin\n")
            f.write(f"\tTemperature = {self.oxidizer_inlet.temperature} [K]\n")
            for key, value in self.oxidizer_inlet.composition.items():
                f.write(f"\tMassFraction-{key} = {value}\n")
            f.write(f"end\n")
            f.write(f"\n")
            f.write(f"numOfSpecies = {self.gas.n_species}\n")
            f.write(f"gridPoints = {N_points}\n")
            f.write(f"\n")

            # Write flame profiles
            f.write(f"body\n")
            write_array(f, "Z", self.flame.mixture_fraction("Bilger"))
            write_array(f, "temperature [K]", self.flame.T)
            for k in range(self.gas.n_species):
                write_array(f, f"massfraction-{self.gas.species_names[k]}", self.flame.Y[k, :])
            write_array(f, "W", self.flame.mean_molecular_weight)
            write_array(f, "ZBilger", self.flame.mixture_fraction("Bilger"))
            write_array(f, "chi [1/s]", solution["chi"])
            write_array(f, "density", self.flame.density)
            write_array(f, "lambda [W/m K]", self.flame.thermal_conductivity)
            write_array(f, "cp [J/kg K]", self.flame.cp_mass)
            write_array(
                f,
                "lambdaOverCp [kg/ms]",
                (self.flame.thermal_conductivity / self.flame.cp_mass),
            )
            write_array(f, "mu [kg/sm]", self.flame.viscosity)
            for k in range(self.gas.n_species):
                write_array(
                    f,
                    f"ProdRate-{self.gas.species_names[k]} [kg/m^3s]",
                    self.flame.net_production_rates[k, :] * self.gas.molecular_weights[k],
                )
            write_array(f, "ProgVar", self._compute_progress_variable())
            write_array(f, "ProdRateProgVar [kg/m^3s]", self._compute_progress_variable_production())
            write_array(f, "TotalEnthalpy [J/kg]", self.flame.enthalpy_mass)
            write_array(f, "HeatRelease [J/m^3 s]", self.flame.heat_release_rate)
            write_array(f, "Diffusivity [m/s]", np.zeros_like(self.flame.grid))  # Real gas thing, zeros are fine for now
            write_array(f, "rho_Diffusivity [kg/(m^2 s)]", np.zeros_like(self.flame.grid))  # Real gas thing, zeros are fine for now
            write_array(f, "TotalEnthalpy_rg [J/kg]", self.flame.enthalpy_mass + self.flame.cp_mass)  # Real gas thing, zeros are fine for now
            write_array(f, "IsoCompressibility [m^2/N]", np.zeros_like(self.flame.grid))  # Real gas thing, zeros are fine for now
            write_array(f, "IsenCompressibility [m^2/N]", np.zeros_like(self.flame.grid))  # Real gas thing, zeros are fine for now
            write_array(f, "SpeedOfSound [m/s]", self.flame.sound_speed)
            write_array(f, "dLnCpDZ", self._compute_dLnCpDZ())
            write_array(f, "SumCpiOverCpMixDYiDZ", np.zeros_like(self.flame.grid))  # Real gas thing, zeros are fine for now

            # Write footer
            # (These are the lewis numbers, which we'll set to 1 here regardless of self.transport_model)
            f.write(f"trailer\n")
            for k in range(self.gas.n_species):
                f.write(f"{self.gas.species_names[k]}\t1\n")
        # fmt: on

    def assemble_FPV_table_CharlesX(
        self,
        output_dir: Path,
        dims: Tuple[int, int, int] = (100, 2, 100),
        force_monotonicity: bool = False,
        igniting_table: bool = False,
        include_species_mass_fractions: Union[str, List[str]] = [],
        include_species_production_rates: Union[str, List[str]] = [],
        include_energy_enthalpy_components: bool = False,
        n_workers: Optional[int] = None,
    ) -> None:
        """Assemble a Flamelet Progress Variable (FPV) table and write it in CharlesX format.

        Args:
            output_dir: Directory path to save the CharlesX table files
            dims: Tuple of (Z, Q, L) dimensions for the table
            force_monotonicity: Whether to force monotonicity in the table
            igniting_table: Whether to assemble an igniting table
            include_species_mass_fractions: List of species for which to include mass fractions
            include_species_production_rates: List of species for which to include production rates
            include_energy_enthalpy_components: Whether to include energy and enthalpy components
        """
        # Build filename
        fuel_str = "".join([key for key in self.fuel_inlet.composition.keys()])
        oxidizer_str = "".join([key for key in self.oxidizer_inlet.composition.keys()])
        pressure_str = f"p{(self.pressure/1e5):04.1f}".replace(".", "_")
        tf_str = f"tf{self.fuel_inlet.temperature:04.0f}"
        to_str = f"to{self.oxidizer_inlet.temperature:04.0f}"
        dim_str = f"{dims[0]}x{dims[1]}x{dims[2]}"
        filename = output_dir / f"{fuel_str}_{oxidizer_str}_{pressure_str}_{tf_str}_{to_str}_{dim_str}.h5"

        # Create the dimensions
        N_sol = len(self.solutions)
        i_cut = dims[0] // 3
        Z = coordinate.CoordinateLinearThenStretched("Z", 0, self.Z_st, 1, i_cut, dims[0] - i_cut)
        Q = coordinate.CoordinatePowerLaw("Sz", 0, 1, dims[1], 2.7)
        L = coordinate.CoordinateLinear("C", 0, 1, dims[2])
        self.table_coords = [Z, Q, L]

        # Handle species and production rates
        if include_species_mass_fractions == "all":
            include_species_mass_fractions = self.gas.species_names
        if include_species_production_rates == "all":
            include_species_production_rates = self.gas.species_names

        # Create the data arrays
        vars = [
            "ROM",
            "T0",
            "E0",
            "GAMMA0",
            "AGAMMA",
            "MU0",
            "AMU",
            "LOC0",
            "ALOC",
            "SRC_PROG",
            "PROG",
            "HeatRelease",
            "rho0",
        ]
        vars += ["ZBilger"]
        vars += include_species_mass_fractions
        vars += ["SRC_" + sp for sp in include_species_production_rates]
        if include_energy_enthalpy_components:
            vars += ["E_CHEM", "E0_SENS", "H0", "H0_SENS"]
        vars += ["TA"]
        data_interp_Z = {var: np.zeros((dims[0], N_sol)) for var in vars}

        # Extract all needed numpy arrays from each flamelet state up-front (serial,
        # cheap) so that subprocess workers receive only plain numpy dicts and a
        # mechanism file path — no Cantera objects cross the process boundary.
        self.logger.info("Extracting flamelet state arrays...")
        flamelet_arrays = []
        for sol in self.solutions:
            self.flame.from_array(sol["state"])
            flamelet_arrays.append({
                "T":                   self.flame.T.copy(),
                "Y":                   self.flame.Y.copy(),
                "P":                   float(self.flame.P),
                "density":             self.flame.density.copy(),
                "viscosity":           self.flame.viscosity.copy(),
                "thermal_conductivity": self.flame.thermal_conductivity.copy(),
                "cp_mass":             self.flame.cp_mass.copy(),
                "int_energy_mass":     self.flame.int_energy_mass.copy(),
                "heat_release_rate":   self.flame.heat_release_rate.copy(),
                "net_production_rates": self.flame.net_production_rates.copy(),
                "enthalpy_mass":       self.flame.enthalpy_mass.copy(),
                "mean_molecular_weight": self.flame.mean_molecular_weight.copy(),
            })

        worker_args = [
            (flamelet_arrays[i], self.solutions[i]["Z"], Z.grid,
             self.mechanism_file, self.prog_def,
             list(include_species_mass_fractions),
             list(include_species_production_rates),
             include_energy_enthalpy_components)
            for i in range(N_sol)
        ]

        self.logger.info(
            f"Processing {N_sol} flamelets in parallel (n_workers={n_workers})..."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_flamelet_worker, worker_args))

        for i, res in enumerate(results):
            for var in vars:
                data_interp_Z[var][:, i] = res[var]

        # Sort flamelets by peak progress variable
        C_peak = np.max(data_interp_Z["PROG"], axis=0)
        i_sort = np.argsort(C_peak)
        for var in vars:
            data_interp_Z[var] = data_interp_Z[var][:, i_sort]

        # Compute the normalized progress variable (Lambda)
        C_arr = data_interp_Z["PROG"]

        if force_monotonicity:
            # Min and max values are taken from the first and last solutions,
            # which have been sorted by peak progress variable
            C_min = np.tile(C_arr[:, 0][:, np.newaxis], (1, N_sol))
            C_max = np.tile(C_arr[:, -1][:, np.newaxis], (1, N_sol))
            L_arr = (C_arr - C_min) / (C_max - C_min)

            # Handle near Z=0 and Z=1, where min and max are close
            tol = 1e-4
            index = np.logical_or(((C_max - C_min) < tol), (C_min > C_max))
            L_uniform = np.tile(np.linspace(0, 1, N_sol), (dims[0], 1))
            L_arr[index] = L_uniform[index]

            # Make CA well-behaved
            tol = 1e-10
            adj = 1e-8
            for i_Z in range(dims[0]):
                dL_min = np.inf
                L_min_adj = np.inf
                for i_L in range(N_sol - 2, 0, -1):
                    dL = L_arr[i_Z, i_L + 1] - L_arr[i_Z, i_L]
                    if dL < tol:
                        dL_min = min(dL_min, dL)
                        L_min_adj = min(L_min_adj, L_arr[i_Z, i_L])
                        L_arr[i_Z, i_L] = L_arr[i_Z, i_L + 1] - adj
                if dL_min < np.inf:
                    self.logger.warning(f"Adjusted Lambda by at least {-dL_min:.2e} at Z = {Z.grid[i_Z]:.2e}")
        else:
            # Min and max values are taken from any solution, and the solution
            # from which they are taken may vary with Z
            C_min = np.tile(np.min(C_arr, axis=1)[:, np.newaxis], (1, N_sol))
            C_max = np.tile(np.max(C_arr, axis=1)[:, np.newaxis], (1, N_sol))
            L_arr = (C_arr - C_min) / (C_max - C_min)

            # Handle near Z=0 and Z=1, where min and max are close
            tol = 1e-4
            index = (C_max - C_min) < tol
            L_uniform = np.tile(np.linspace(0, 1, N_sol), (dims[0], 1))
            L_arr[index] = L_uniform[index]

            # Treat values near max
            L_arr[L_arr > 1.0 - tol] = 1.0

        data_interp_Z["PROG_NORM"] = L_arr

        # Interpolate the data onto the Lambda coordinate
        vars_interp = vars + ["PROG_NORM"]
        self.data_table = {var: np.zeros((dims[0], dims[1], dims[2])) for var in vars_interp}
        for var in vars_interp:
            for i in range(dims[0]):
                interp = interpolate.interp1d(
                    data_interp_Z["PROG_NORM"][i, :],
                    data_interp_Z[var][i, :],
                    axis=0,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                data_table_i = interp(L.grid)
                data_table_i = np.tile(data_table_i, (dims[1], 1))  # Expand to cover Q dimension
                self.data_table[var][i, :, :] = data_table_i
        
        # Treat activation temperature
        self.data_table["TA"][self.data_table["TA"] < 0.0] = 0.0
        self.data_table["TA"][np.isnan(self.data_table["TA"])] = 0.0

        # Handle ignition
        tol = 1e-8
        self.data_table["SRC_PROG"][self.data_table["PROG_NORM"] > 1.0 - tol] = 0.0
        if not igniting_table:
            self.data_table["SRC_PROG"][self.data_table["PROG_NORM"] < tol] = 0.0
        
        # Write the table to the HDF5 file in CharlesX format
        with h5py.File(filename, "w") as f:
            # Create variable-length string dtype with ASCII encoding and space padding
            str_type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
            str_type_id.set_size(h5py.h5t.VARIABLE)
            str_type_id.set_cset(h5py.h5t.CSET_ASCII)
            str_type_id.set_strpad(h5py.h5t.STR_SPACEPAD)
            str_dtype = h5py.Datatype(str_type_id)
            
            # Header group
            header = f.create_group("Header")

            # Doubles subgroup
            doubles = header.create_group("Doubles")
            doubles.attrs["Number of doubles"] = [np.int32(2)]
            double_0 = doubles.create_dataset("Double_0", data=["Reference Pressure"], dtype=str_dtype)
            double_0.attrs["Value"] = [np.float64(self.pressure)]
            double_1 = doubles.create_dataset("Double_1", data=["Version"], dtype=str_dtype)
            double_1.attrs["Value"] = [np.float64(0.2)]
            
            # Strings subgroup
            strings = header.create_group("Strings")
            strings.attrs["Number of strings"] = [np.int32(2)]
            strings.create_dataset("String_0", data=["Combustion Model", "FPVA"], dtype=str_dtype)
            strings.create_dataset("String_1", data=["Table Type", "COEFF"], dtype=str_dtype)
            
            # Header datasets
            header.create_dataset("Number of dimensions", data=[3], dtype=np.int32)
            header.create_dataset("Number of variables", data=[len(vars)], dtype=np.int32)
            header.create_dataset("Variable Names", data=vars, dtype=str_dtype)

            # Data dataset
            n_tot = dims[0] * dims[1] * dims[2]
            data_raw = np.empty((n_tot * len(vars)), dtype=np.float32)
            for i, var in enumerate(vars):
                data_raw[i * n_tot : (i + 1) * n_tot] = self.data_table[var].ravel(order="C")
            f.create_dataset("Data", data=data_raw, dtype=np.float32)

            # Coordinates group
            coords = f.create_group("Coordinates")
            Z.write_hdf5(coords, "Coor_0")
            Q.write_hdf5(coords, "Coor_1")
            L.write_hdf5(coords, "Coor_2")

    def learn_strain_chi_st_mapping(self, output_file: Optional[str] = None):
        """Learn a mapping between strain rate and scalar dissipation rate at stoichiometry.

        Args:
            output_file: Optional file path to save the mapping as a JSON file

        Note:
            This method uses the computed solutions to learn a mapping between strain rate
            and scalar dissipation rate at the stoichiometric mixture fraction. The mapping
            is learned using a simple linear regression model.
        """
        from scipy.stats import linregress

        X = np.array([np.log10(sol["metadata"]["strain_rate_nom"]) for sol in self.solutions]).reshape(-1, 1)
        y = np.array([np.log10(sol["metadata"]["chi_st"]) for sol in self.solutions])
        slope, intercept, r_value, p_value, std_err = linregress(X.ravel(), y)
        self.strain_chi_st_model_params = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }

        if output_file is not None:
            with open(output_file, "w") as f:
                json.dump(self.strain_chi_st_model_params, f)

    def plot_s_curve(
        self,
        x_quantity: Literal["strain_rate_max", "chi_st"] = "chi_st",
        y_quantity: Literal["T_max", "T_st"] = "T_max",
        output_file: Optional[str] = None,
    ):
        """Plot the S-curve with configurable axes.

        Args:
            x_quantity: Which quantity to plot on x-axis ('strain_rate_max' or 'chi_st')
            y_quantity: Which quantity to plot on y-axis ('T_max' or 'T_st')
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt

        plt.rcParams.update(pyplot_params)

        x_values = [d["metadata"][x_quantity] for d in self.solutions]
        y_values = [d["metadata"][y_quantity] for d in self.solutions]

        # Set up axis labels
        x_labels = {"strain_rate_max": r"$\alpha$ [1/s]", "chi_st": r"$\chi_{st}$ [1/s]"}

        y_labels = {"T_max": r"$T_\textrm{max}$ [K]", "T_st": r"$T_{st}$ [K]"}

        fig, ax = plt.subplots()
        ax.semilogx(x_values, y_values, "o-")
        ax.set_xlabel(x_labels[x_quantity])
        ax.set_ylabel(y_labels[y_quantity])
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_temperature_profiles(
        self,
        output_file: Optional[str] = None,
        num_profiles: Optional[int] = None,
        colormap: str = "viridis",
    ):
        """Plot temperature vs mixture fraction profiles for all solutions.

        Creates a plot showing temperature profiles colored by scalar dissipation rate.

        Args:
            output_file: Optional file path to save the figure
            num_profiles: Optional number of profiles to plot (will sample evenly)
            colormap: Name of colormap to use for the profiles

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        plt.rcParams.update(pyplot_params)

        # Select which solutions to plot
        if num_profiles is None:
            solutions_to_plot = self.solutions
        else:
            indices = np.linspace(0, len(self.solutions) - 1, num_profiles, dtype=int)
            solutions_to_plot = [self.solutions[i] for i in indices]

        fig, ax = plt.subplots()

        # Create log-scaled colormap based on chi_st values
        chi_st_values = [sol["metadata"]["chi_st"] for sol in solutions_to_plot]
        norm = LogNorm(vmin=min(chi_st_values), vmax=max(chi_st_values))
        cmap = plt.get_cmap(colormap)

        # Plot each profile
        for solution in solutions_to_plot:
            color = cmap(norm(solution["metadata"]["chi_st"]))
            ax.plot(solution["Z"], solution["state"].T, color=color, alpha=0.7)

        # Add colorbar with log scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\chi_{st}$ [1/s]")

        # Add stoichiometric mixture fraction line
        ax.axvline(self.Z_st, color="k", linestyle="--", alpha=0.3)

        ax.set_xlabel(r"$Z$ [-]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_progress_variable_profiles(
        self,
        output_file: Optional[str] = None,
        num_profiles: Optional[int] = None,
        colormap: str = "viridis",
    ):
        """Plot progress variable vs mixture fraction profiles for all solutions.

        Creates a plot showing progress variable profiles colored by scalar dissipation rate.

        Args:
            output_file: Optional file path to save the figure
            num_profiles: Optional number of profiles to plot (will sample evenly)
            colormap: Name of colormap to use for the profiles

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        plt.rcParams.update(pyplot_params)

        # Select which solutions to plot
        if num_profiles is None:
            solutions_to_plot = self.solutions
        else:
            indices = np.linspace(0, len(self.solutions) - 1, num_profiles, dtype=int)
            solutions_to_plot = [self.solutions[i] for i in indices]

        fig, ax = plt.subplots()

        # Create log-scaled colormap based on chi_st values
        chi_st_values = [sol["metadata"]["chi_st"] for sol in solutions_to_plot]
        norm = LogNorm(vmin=min(chi_st_values), vmax=max(chi_st_values))
        cmap = plt.get_cmap(colormap)

        # Plot each profile
        for solution in solutions_to_plot:
            self.flame.from_array(solution["state"])
            C = self._compute_progress_variable()
            color = cmap(norm(solution["metadata"]["chi_st"]))
            ax.plot(solution["Z"], C, color=color, alpha=0.7)

        # Add colorbar with log scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(r"$\chi_{st}$ [1/s]")

        # Add stoichiometric mixture fraction line
        ax.axvline(self.Z_st, color="k", linestyle="--", alpha=0.3)

        ax.set_xlabel(r"$Z$ [-]")
        ax.set_ylabel(r"$C$ [-]")
        ax.grid(True, alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_current_state(self, output_file: Optional[str] = None):
        """Plot the current state of the flame in physical space.

        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt

        plt.rcParams.update(pyplot_params)

        if self.flame is None:
            raise ValueError("Flame has not been initialized. Please initialize the flame first.")

        fig, ax = plt.subplots()
        ax.plot(self.flame.grid, self.flame.T, color="red")
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_current_state_composition_space(self, output_file: Optional[str] = None):
        """Plot the current state of the flame in composition space.

        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt

        plt.rcParams.update(pyplot_params)

        if self.flame is None:
            raise ValueError("Flame has not been initialized. Please initialize the flame first.")

        Z = self.flame.mixture_fraction("Bilger")
        T = self.flame.T

        fig, ax = plt.subplots()
        ax.plot(Z, T, color="red")
        ax.axvline(self.Z_st, color="k", linestyle="--")
        ax.set_xlabel(r"$Z$ [-]")
        ax.set_ylabel(r"$T$ [K]")
        ax.grid(True, alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_strain_chi_st(self, strain_rate_type: Literal["max", "nom"] = "nom", output_file: Optional[str] = None):
        """Plot the strain rate vs scalar dissipation rate at stoichiometric mixture fraction.

        Args:
            output_file: Optional file path to save the figure

        Returns:
            Tuple[Figure, Axes]: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt

        plt.rcParams.update(pyplot_params)

        strain_rates = [sol["metadata"][f"strain_rate_{strain_rate_type}"] for sol in self.solutions]
        chi_st_values = [sol["metadata"]["chi_st"] for sol in self.solutions]

        fig, ax = plt.subplots()
        ax.scatter(chi_st_values, strain_rates)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\chi_{st}$ [1/s]")
        ax.set_ylabel(r"$\alpha$ [1/s]")
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.set_axisbelow(True)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        if output_file:
            fig.savefig(output_file, bbox_inches="tight", dpi=300)

        return fig, ax

    def plot_flamelets_interactive(
        self,
        output_dir: Optional[Path] = None,
        save_filtered: bool = True,
        filtered_filename: str = "solutions_filtered.h5",
    ) -> Dict[str, List[int]]:
        """Interactively inspect flamelet solutions and select ones to omit.

        Opens a matplotlib GUI that plots every flamelet solution and lets the user pick
        the x/y variables. Two view modes are available:

        - **Scatter**: one point per flamelet, with axes drawn from per-flamelet scalar
          quantities (the ``metadata`` keys, e.g. ``chi_st`` vs ``T_max``). Best for
          spotting outliers on the S-curve.
        - **Profiles**: one curve per flamelet over its grid, with axes drawn from profile
          variables (``Z``, ``T``, ``C``, ``chi``, and selected species mass fractions).

        Click a point (scatter) or curve (profiles) to toggle whether that flamelet is
        omitted. Omitted flamelets are greyed out. When "Save & Close" is pressed and at
        least one flamelet is omitted, the kept flamelets are written to a filtered HDF5
        solutions file (contiguously renumbered) that ``load_solutions`` can read back.

        Args:
            output_dir: Directory to write the filtered solutions file (defaults to the
                current working directory).
            save_filtered: Whether to write the filtered file on save.
            filtered_filename: Name of the filtered solutions file to write.

        Returns:
            Dict with keys ``"kept"`` and ``"omitted"`` listing the original solution
            indices that were kept / omitted.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, Button, TextBox
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        n_sol = len(self.solutions)
        if n_sol == 0:
            raise ValueError("No solutions to plot.")

        # Disable LaTeX rendering for this interactive session: variable names contain
        # underscores and button labels contain "&", both of which break the LaTeX
        # backend. Restored after the (blocking) window closes.
        prev_usetex = plt.rcParams["text.usetex"]
        plt.rcParams["text.usetex"] = False

        # --- Variable registries -------------------------------------------------------
        # Scalar variables (scatter view): one value per flamelet. Only metadata keys that
        # are present in *every* solution are offered, so querying never hits a KeyError
        # (different branches store slightly different metadata, e.g. the extinct branch
        # omits "Tc_increment"/"errors").
        common_meta = set.intersection(*[set(sol["metadata"].keys()) for sol in self.solutions])
        scalar_keys = ["index"] + [
            k for k, v in self.solutions[0]["metadata"].items()
            if k in common_meta and isinstance(v, (int, float, np.number))
        ]

        def scalar_values(key: str) -> np.ndarray:
            if key == "index":
                return np.arange(n_sol, dtype=float)
            return np.array([float(sol["metadata"].get(key, np.nan)) for sol in self.solutions])

        # Profile variables (profiles view): an array per flamelet over the grid. Available
        # names are the mixture fraction (Z), temperature (T), progress variable (C), the
        # progress-variable source term (SRC_C), and any species mass fraction.
        #
        # Values are read directly from each solution's stored ``SolutionArray`` (T and Y
        # are already in memory) rather than restoring every flamelet into the 1D flame
        # domain via ``from_array`` -- the latter is ~20x slower and dominated the load
        # time. Note ``SolutionArray.Y`` is shaped (n_points, n_species).
        species_names = list(self.gas.species_names)
        profile_keys = ["Z", "T", "C", "SRC_C"] + species_names
        prog_idx = np.array([self.gas.species_index(s) for s in self.prog_def])
        prog_coef = np.array([self.prog_def[s] for s in self.prog_def])
        mol_weights = self.gas.molecular_weights

        base_cache: Dict[int, Dict[str, np.ndarray]] = {}
        srcC_cache: Dict[int, np.ndarray] = {}

        def profile_base(i: int) -> Dict[str, np.ndarray]:
            if i not in base_cache:
                sol = self.solutions[i]
                st = sol["state"]
                base_cache[i] = {
                    "Z": np.asarray(sol["Z"]),
                    "T": np.asarray(st.T),
                    "Y": np.asarray(st.Y),  # (n_points, n_species)
                }
            return base_cache[i]

        def profile_value(i: int, key: str) -> np.ndarray:
            b = profile_base(i)
            if key in ("Z", "T"):
                return b[key]
            if key == "C":
                return b["Y"][:, prog_idx] @ prog_coef
            if key == "SRC_C":
                if i not in srcC_cache:
                    # Net production rate is a kinetic quantity, so it is computed on
                    # demand (only when SRC_C is actually requested) and cached.
                    wdot = np.asarray(self.solutions[i]["state"].net_production_rates)
                    srcC_cache[i] = (wdot[:, prog_idx] * mol_weights[prog_idx]) @ prog_coef
                return srcC_cache[i]
            return b["Y"][:, self.gas.species_index(key)]

        # Per-flamelet branch colors (used to color profile curves so outliers / bad
        # branches stand out).
        branch_ids = [int(self.solutions[i]["metadata"].get("branch_id", 0)) for i in range(n_sol)]
        unique_branches = sorted(set(branch_ids))
        _cmap = plt.get_cmap("tab10")
        branch_color = {b: _cmap(k % 10) for k, b in enumerate(unique_branches)}

        def parse_index_list(text: str) -> set:
            """Parse a comma-separated list of indices/ranges, e.g. '1, 5, 10-20'."""
            out: set = set()
            for tok in text.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok.lstrip("-"):  # a range like 10-20 (not a bare negative)
                    a, b = tok.split("-", 1)
                    lo, hi = int(a), int(b)
                    out.update(range(min(lo, hi), max(lo, hi) + 1))
                else:
                    out.add(int(tok))
            return {i for i in out if 0 <= i < n_sol}

        # Variables that read best on a log axis.
        log_vars = {"chi_st", "strain_rate_max", "strain_rate_nom",
                    "total_heat_release_rate"}

        # --- State ---------------------------------------------------------------------
        omitted: set = set()
        # X/Y selections are remembered separately per view mode.
        scatter_x = "chi_st" if "chi_st" in scalar_keys else (scalar_keys[1] if len(scalar_keys) > 1 else scalar_keys[0])
        scatter_y = "T_max" if "T_max" in scalar_keys else scalar_keys[-1]
        state = {
            "mode": "Scatter",
            "Scatter": {"x": scatter_x, "y": scatter_y},
            "Profiles": {"x": "Z", "y": "C"},
        }
        artist_to_index: Dict[object, int] = {}

        def cur() -> Dict[str, str]:
            return state[state["mode"]]

        # --- Figure layout -------------------------------------------------------------
        fig = plt.figure(figsize=(13, 7))
        fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.10)
        ax = fig.add_subplot(111)

        ax_mode = fig.add_axes([0.02, 0.78, 0.22, 0.15])
        ax_xradio = fig.add_axes([0.02, 0.44, 0.10, 0.30])
        ax_yradio = fig.add_axes([0.14, 0.44, 0.10, 0.30])
        ax_omit = fig.add_axes([0.075, 0.30, 0.165, 0.05])
        ax_invert = fig.add_axes([0.02, 0.22, 0.10, 0.05])
        ax_reset = fig.add_axes([0.14, 0.22, 0.10, 0.05])
        ax_save = fig.add_axes([0.02, 0.14, 0.22, 0.05])
        for a in (ax_xradio, ax_yradio):
            a.set_title("")

        mode_radio = RadioButtons(ax_mode, ("Scatter", "Profiles"), active=0)
        ax_mode.set_title("View mode", fontsize=9)
        omit_box = TextBox(ax_omit, "Omit ", initial="")
        ax_omit.set_title("Omit indices (e.g. 1, 5, 10-20)", fontsize=7, loc="left")
        invert_btn = Button(ax_invert, "Invert")
        reset_btn = Button(ax_reset, "Reset")
        save_btn = Button(ax_save, "Save & Close")

        info = fig.text(0.30, 0.965, "", fontsize=9, va="bottom")

        # Holders for the X/Y selector widgets (rebuilt when the mode changes). In Scatter
        # mode these are RadioButtons over the (small) scalar key list; in Profiles mode,
        # where there are 50+ possible variables, they are typed TextBoxes validated
        # against ``profile_keys``.
        widgets = {"x": None, "y": None}

        def build_var_radios():
            # Fully disconnect the previous widgets before clearing their axes; otherwise
            # their blitting draw_event callbacks keep firing on stale artists and crash.
            for key in ("x", "y"):
                if widgets[key] is not None:
                    widgets[key].disconnect_events()
            ax_xradio.clear()
            ax_yradio.clear()
            sel = cur()

            if state["mode"] == "Scatter":
                # Tall side-by-side radio columns.
                ax_xradio.set_position([0.02, 0.44, 0.10, 0.30])
                ax_yradio.set_position([0.14, 0.44, 0.10, 0.30])
                keys = scalar_keys
                if sel["x"] not in keys:
                    sel["x"] = keys[0]
                if sel["y"] not in keys:
                    sel["y"] = keys[min(1, len(keys) - 1)]
                ax_xradio.set_title("X", fontsize=9)
                ax_yradio.set_title("Y", fontsize=9)
                widgets["x"] = RadioButtons(ax_xradio, keys, active=keys.index(sel["x"]))
                widgets["y"] = RadioButtons(ax_yradio, keys, active=keys.index(sel["y"]))
                for labels in (widgets["x"].labels, widgets["y"].labels):
                    for lbl in labels:
                        lbl.set_fontsize(8)
                widgets["x"].on_clicked(on_x)
                widgets["y"].on_clicked(on_y)
            else:
                # Short, stacked text-entry boxes; typing a name and pressing Enter
                # updates the axis (valid: Z, T, C, SRC_C, or any species name).
                ax_xradio.set_position([0.04, 0.66, 0.20, 0.045])
                ax_yradio.set_position([0.04, 0.56, 0.20, 0.045])
                ax_xradio.set_title("X  (Z, T, C, SRC_C, <species>)", fontsize=7, loc="left")
                ax_yradio.set_title("Y", fontsize=7, loc="left")
                widgets["x"] = TextBox(ax_xradio, "", initial=sel["x"])
                widgets["y"] = TextBox(ax_yradio, "", initial=sel["y"])
                widgets["x"].on_submit(lambda text: on_text("x", text))
                widgets["y"].on_submit(lambda text: on_text("y", text))

        def redraw():
            ax.clear()
            artist_to_index.clear()
            sel = cur()
            xkey, ykey = sel["x"], sel["y"]

            if state["mode"] == "Scatter":
                xv = scalar_values(xkey)
                yv = scalar_values(ykey)
                kept_mask = np.array([i not in omitted for i in range(n_sol)])
                if kept_mask.any():
                    sc = ax.scatter(xv[kept_mask], yv[kept_mask],
                                    c="tab:blue", picker=True, zorder=3, s=40)
                    artist_to_index[sc] = np.nonzero(kept_mask)[0]
                if (~kept_mask).any():
                    sc_om = ax.scatter(xv[~kept_mask], yv[~kept_mask],
                                       facecolors="none", edgecolors="0.6",
                                       picker=True, zorder=2, s=40)
                    artist_to_index[sc_om] = np.nonzero(~kept_mask)[0]
            else:
                # All curves are drawn as one (kept) / two (kept + omitted) LineCollections
                # rather than thousands of Line2D artists, which keeps both the initial
                # draw and click-picking fast.
                kept_segs, kept_idx, kept_colors = [], [], []
                om_segs, om_idx = [], []
                for i in range(n_sol):
                    seg = np.column_stack([profile_value(i, xkey), profile_value(i, ykey)])
                    if i in omitted:
                        om_segs.append(seg)
                        om_idx.append(i)
                    else:
                        kept_segs.append(seg)
                        kept_idx.append(i)
                        kept_colors.append(branch_color[branch_ids[i]])
                if om_segs:
                    lc_om = LineCollection(om_segs, colors="0.8", linewidths=0.6,
                                           alpha=0.5, zorder=1, picker=True)
                    lc_om.set_pickradius(4)
                    ax.add_collection(lc_om, autolim=True)
                    artist_to_index[lc_om] = np.array(om_idx)
                if kept_segs:
                    lc = LineCollection(kept_segs, colors=kept_colors, linewidths=0.8,
                                        alpha=0.8, zorder=2, picker=True)
                    lc.set_pickradius(4)
                    ax.add_collection(lc, autolim=True)
                    artist_to_index[lc] = np.array(kept_idx)
                ax.autoscale_view()
                if len(unique_branches) > 1:
                    ax.legend(handles=[Line2D([0], [0], color=branch_color[b], lw=1.5,
                                              label=f"branch {b}") for b in unique_branches],
                              fontsize=8, loc="best")

            ax.set_xlabel(xkey)
            ax.set_ylabel(ykey)
            ax.set_xscale("log" if xkey in log_vars else "linear")
            ax.set_yscale("log" if ykey in log_vars else "linear")
            ax.grid(True, which="both", alpha=0.2)
            ax.set_title(f"{len(omitted)} omitted / {n_sol} flamelets "
                         f"(click a curve/point to toggle, or use the Omit box)", fontsize=10)
            fig.canvas.draw_idle()

        # Keep the "Omit indices" text box in sync with the omitted set without
        # re-triggering its own submit callback.
        omit_suppress = {"v": False}

        def refresh_omit_box():
            omit_suppress["v"] = True
            omit_box.set_val(", ".join(str(i) for i in sorted(omitted)))
            omit_suppress["v"] = False

        def toggle(i: int):
            if i in omitted:
                omitted.discard(i)
            else:
                omitted.add(i)
            meta = self.solutions[i]["metadata"]
            info.set_text(
                f"Flamelet {i}: chi_st={meta.get('chi_st', float('nan')):.3e}, "
                f"T_max={meta.get('T_max', float('nan')):.1f} K, "
                f"branch={meta.get('branch_id', '?')}  "
                f"[{'OMITTED' if i in omitted else 'kept'}]"
            )
            refresh_omit_box()
            redraw()

        def on_pick(event):
            idx = artist_to_index.get(event.artist)
            if idx is None:
                return
            if np.ndim(idx) == 0:  # a single Line2D -> one flamelet index
                toggle(int(idx))
            else:  # a scatter collection -> map picked offset to flamelet index
                if len(event.ind) == 0:
                    return
                toggle(int(idx[event.ind[0]]))

        def on_x(label):
            cur()["x"] = label
            redraw()

        def on_y(label):
            cur()["y"] = label
            redraw()

        reverting = {"x": False, "y": False}

        def on_text(axis, text):
            if reverting[axis]:
                return
            key = text.strip()
            valid = key in profile_keys or key in self.gas.species_names
            if not valid:
                info.set_text(f"Unknown profile variable '{key}'. "
                              f"Use Z, T, C, SRC_C, or a species name.")
                # Revert the box to the last valid selection (guarded re-entry).
                reverting[axis] = True
                widgets[axis].set_val(cur()[axis])
                reverting[axis] = False
                fig.canvas.draw_idle()
                return
            info.set_text("")
            cur()[axis] = key
            redraw()

        def on_mode(label):
            state["mode"] = label
            build_var_radios()
            redraw()

        def on_omit_submit(text):
            if omit_suppress["v"]:
                return
            try:
                new = parse_index_list(text)
            except ValueError:
                info.set_text("Could not parse omit list. Use e.g. '1, 5, 10-20'.")
                refresh_omit_box()
                fig.canvas.draw_idle()
                return
            omitted.clear()
            omitted.update(new)
            info.set_text(f"Omit list set: {len(omitted)} flamelet(s).")
            refresh_omit_box()
            redraw()

        def on_invert(_event):
            omitted.symmetric_difference_update(range(n_sol))
            refresh_omit_box()
            redraw()

        def on_reset(_event):
            omitted.clear()
            info.set_text("")
            refresh_omit_box()
            redraw()

        result: Dict[str, List[int]] = {
            "kept": list(range(n_sol)),
            "omitted": [],
        }

        def on_save(_event):
            kept = [i for i in range(n_sol) if i not in omitted]
            result["kept"] = kept
            result["omitted"] = sorted(omitted)
            if save_filtered and omitted:
                out = Path(output_dir) if output_dir is not None else Path.cwd()
                out.mkdir(parents=True, exist_ok=True)
                self._save_solution_subset(out, kept, filtered_filename)
                self.logger.info(
                    f"Wrote {len(kept)} kept flamelets to {out / filtered_filename} "
                    f"(omitted {len(omitted)}: {sorted(omitted)})"
                )
            elif save_filtered:
                self.logger.info("No flamelets omitted; filtered file not written.")
            plt.close(fig)

        mode_radio.on_clicked(on_mode)
        omit_box.on_submit(on_omit_submit)
        invert_btn.on_clicked(on_invert)
        reset_btn.on_clicked(on_reset)
        save_btn.on_clicked(on_save)
        fig.canvas.mpl_connect("pick_event", on_pick)

        build_var_radios()
        redraw()
        try:
            plt.show()
        finally:
            plt.rcParams["text.usetex"] = prev_usetex

        return result

    def plot_table(
        self,
        output_prefix: str,
        vars: List[str],
        Q_plot: float = 0.0,
        colormap: str = "viridis",
    ):
        """Plot a variable from the FPV table.

        Args:
            output_prefix: Prefix for the output files
            vars: List of variables to plot
            Q_plot: Value of mixture fraction variance slice to plot
            colormap: Name of colormap to use for the plots
        """
        import matplotlib.pyplot as plt
        from scipy.interpolate import RegularGridInterpolator

        plt.rcParams.update(pyplot_params)

        if vars == "all":
            vars = list(self.data_table.keys())

        for var in vars:
            if Q_plot == 0.0:
                plot_data = self.data_table[var][:, 0, :].T
            elif Q_plot == 1.0:
                plot_data = self.data_table[var][:, -1, :].T
            else:
                interpolator = RegularGridInterpolator(
                    [self.table_coords[i].grid for i in range(len(self.table_coords))],
                    self.data_table[var][:, :, :].T,
                    bounds_error=False,
                    fill_value=None,
                )
                Z_plot, L_plot = np.meshgrid(self.table_coords[0].grid, self.table_coords[2].grid, indexing="ij")
                Q_plot = np.full_like(Z_plot, Q_plot)
                interp_grid = np.array([Z_plot.ravel(), Q_plot.ravel(), L_plot.ravel()]).T
                plot_data = interpolator(interp_grid).reshape(Z_plot.shape)

            fig, ax = plt.subplots()
            ax.set_title(var)
            ax.set_xlabel(r"$Z$ [-]")
            ax.set_ylabel(r"$\Lambda$ [-]")
            ax.set_aspect("equal")
            c = ax.contourf(
                self.table_coords[0].grid,
                self.table_coords[2].grid,
                plot_data,
                levels=100,
                cmap=colormap,
            )
            fig.colorbar(c, ax=ax)
            fig.savefig(f"{output_prefix}_{var}.png", bbox_inches="tight", dpi=300)
