"""
flamelet_data.py
----------------
Builds neural-network training data directly from a raw flamelet solutions
database (the HDF5 file written by ``FlameletTableGenerator.save_all_solutions``,
e.g. ``solutions.h5`` / ``solutions_filtered.h5``).

Each laminar flamelet solution is a 1-D profile in physical space. For every grid
point of every solution we reconstruct the control variables (mixture fraction
``Z``, progress variable ``C``, and an all-zero ``Zv`` placeholder) and the
requested output quantities, then concatenate into a single point cloud
``inputs`` (dict of name -> (N,) arrays) and ``Y`` of shape (N, M).

This reuses ``FlameletTableGenerator``'s own loaders and derived-quantity helpers
so the physics (Bilger mixture fraction, progress-variable definition) stays
consistent with the rest of the pipeline.
"""

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from fpvgen.flamelet_table_generator import FlameletTableGenerator

logger = logging.getLogger(__name__)

# Control variables this loader knows how to produce per grid point.
CONTROL_VARIABLES = ("Z", "Zv", "C")


def _resolve_output_names(outputs: Sequence[str], species_names: List[str]) -> List[str]:
    """Expand convenience tokens (e.g. ``all_species``) into explicit names."""
    resolved: List[str] = []
    for name in outputs:
        if name == "all_species":
            resolved.extend(species_names)
        elif name == "all_source_terms":
            resolved.extend(f"SRC_{sp}" for sp in species_names)
        else:
            resolved.append(name)
    return resolved


def _extract_outputs(gen: FlameletTableGenerator, output_cols: Sequence[str]) -> np.ndarray:
    """Extract the requested output columns for the current ``gen.flame`` state.

    Returns an array of shape (n_points, len(output_cols)).
    """
    flame = gen.flame
    species_names = gen.gas.species_names
    mw = gen.gas.molecular_weights

    cols = []
    for name in output_cols:
        if name == "T":
            col = flame.T
        elif name in ("rho", "density"):
            col = flame.density
        elif name in ("mu", "viscosity"):
            col = flame.viscosity
        elif name in ("lambda", "thermal_conductivity"):
            col = flame.thermal_conductivity
        elif name in ("cp", "cp_mass"):
            col = flame.cp_mass
        elif name in ("HeatRelease", "heat_release_rate"):
            col = flame.heat_release_rate
        elif name in ("PROG", "C"):
            col = gen._compute_progress_variable()
        elif name in ("SRC_PROG", "SRC_C"):
            col = gen._compute_progress_variable_production()
        elif name.startswith("SRC_") and name[4:] in species_names:
            idx = gen.gas.species_index(name[4:])
            col = flame.net_production_rates[idx, :] * mw[idx]
        elif name in species_names:
            idx = gen.gas.species_index(name)
            col = flame.Y[idx, :]
        else:
            raise ValueError(
                f"Unknown output variable '{name}'. Expected 'T', 'rho', 'mu', "
                f"'lambda', 'cp', 'HeatRelease', 'PROG'/'C', 'SRC_PROG'/'SRC_C', "
                f"a species name, or 'SRC_<species>'."
            )
        cols.append(np.asarray(col, dtype=np.float64))
    return np.stack(cols, axis=1)


def load_training_data(
    solutions_file: Path,
    input_names: Sequence[str],
    outputs: Sequence[str],
) -> Tuple[dict, np.ndarray, List[str]]:
    """Load raw flamelet solutions and assemble a NN training point cloud.

    Args:
        solutions_file: Path to the HDF5 solutions database.
        input_names: Ordered control variables to use as NN inputs (subset of
            ``CONTROL_VARIABLES``).
        outputs: Output variable names (may include ``all_species`` /
            ``all_source_terms`` convenience tokens).

    Returns:
        Tuple of:
            - inputs: dict mapping every control-variable name (``Z``, ``Zv``,
              ``C``) to a concatenated (N,) array.
            - Y: concatenated output array of shape (N, M).
            - output_cols: the resolved, explicit list of output names (length M).
    """
    unknown = [n for n in input_names if n not in CONTROL_VARIABLES]
    if unknown:
        raise ValueError(
            f"Unknown input variable(s) {unknown}. Supported: {list(CONTROL_VARIABLES)}."
        )

    logger.info(f"Loading flamelet solutions from {solutions_file}")
    gen = FlameletTableGenerator.load_solutions(str(solutions_file))
    n_sol = len(gen.solutions)
    if n_sol == 0:
        raise ValueError(f"No solutions found in {solutions_file}")

    output_cols = _resolve_output_names(outputs, gen.gas.species_names)
    logger.info(f"Reconstructing {n_sol} flamelets into training samples "
                f"({len(output_cols)} outputs)")

    Z_list, C_list, Y_list = [], [], []
    for sol in gen.solutions:
        gen.flame.from_array(sol["state"])
        Z = np.asarray(sol["Z"], dtype=np.float64)
        C = gen._compute_progress_variable()
        Z_list.append(Z)
        C_list.append(C)
        Y_list.append(_extract_outputs(gen, output_cols))

    Z_all = np.concatenate(Z_list)
    C_all = np.concatenate(C_list)
    inputs = {
        "Z": Z_all,
        "C": C_all,
        "Zv": np.zeros_like(Z_all),  # laminar flamelets carry no variance
    }
    Y = np.concatenate(Y_list, axis=0)

    logger.info(f"Assembled {len(Z_all):,} samples from {n_sol} flamelets")
    return inputs, Y, output_cols
