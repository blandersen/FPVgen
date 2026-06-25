# train_flamelet.py
"""
Command-line interface for training a flamelet neural network from a TOML config.

Trains a fully-connected neural network that maps flamelet control variables
(by default mixture fraction ``Z`` and progress variable ``C``) to thermochemical
state (temperature, species mass fractions, transport properties, source terms).
The network is trained directly from a raw flamelet solutions database produced
by ``generate_table`` -- no assembled table is required.

Usage:
    train_flamelet <config> [--verbose]

Configuration File Format:
    Uses the same TOML file as ``generate_table``. The training run is configured
    with a ``[neural_net]`` section:

        [neural_net]
        solutions_file = "flamelet_results/solutions_filtered.h5"  # optional
        inputs  = ["Z", "C"]                 # control variables (subset of Z, Zv, C)
        outputs = ["T", "rho", "mu", "cp", "SRC_PROG"]  # or "all_species" etc.
        checkpoint = "flamelet_results/flamelet_nn.pt"  # optional
        hidden_dim = 128
        n_layers   = 4
        batch_size = 512
        max_epochs = 3000
        lr         = 1e-3
        val_frac   = 0.15
        patience   = 50
        seed       = 42

    If ``solutions_file`` is omitted it defaults to
    ``<solver.output_dir>/solutions.h5``.
"""
import argparse
import logging
import sys
from pathlib import Path

import tomli
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from fpvgen.flamelet_data import load_training_data
from fpvgen.flamelet_nn import FlametableDataset, FlametableNN

logger = logging.getLogger(__name__)

# Defaults for [neural_net] keys
DEFAULTS = {
    "inputs": ["Z", "C"],
    "outputs": ["T", "rho", "mu", "cp", "SRC_PROG"],
    "hidden_dim": 128,
    "n_layers": 4,
    "batch_size": 512,
    "max_epochs": 3000,
    "lr": 1e-3,
    "val_frac": 0.15,
    "patience": 50,
    "seed": 42,
}


def load_config(config_file: Path) -> dict:
    """Load and validate the TOML configuration file.

    Args:
        config_file: Path to the TOML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        ValueError: If the file cannot be read or the [neural_net] section is missing.
    """
    try:
        with open(config_file, "rb") as f:
            config = tomli.load(f)
    except Exception as e:
        raise ValueError(f"Error reading config file: {e}")

    if "neural_net" not in config:
        raise ValueError("Missing required config section: [neural_net]")

    return config


def resolve_solutions_file(config: dict, config_dir: Path) -> Path:
    """Resolve the solutions database path from the config.

    Uses ``[neural_net].solutions_file`` if given, else
    ``<solver.output_dir>/solutions.h5``. Relative paths are resolved against the
    directory containing the config file.
    """
    nn_cfg = config["neural_net"]
    if "solutions_file" in nn_cfg:
        path = Path(nn_cfg["solutions_file"])
    else:
        output_dir = Path(config.get("solver", {}).get("output_dir", "flamelet_results"))
        path = output_dir / "solutions.h5"

    if not path.is_absolute():
        path = config_dir / path
    return path


def build_loaders(dataset: FlametableDataset, val_frac: float, batch_size: int, seed: int):
    """Split the dataset and build train/val DataLoaders."""
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


def train(model, train_loader, val_loader, params, checkpoint, device):
    """Run the training loop with early stopping; save the best checkpoint."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, params["max_epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(f"epoch {epoch:4d}  train={train_loss:.3e}  "
                        f"val={val_loss:.3e}  lr={lr_now:.1e}")

        # ── checkpoint on best val ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "model_config": model.config,
                "input_names": params["inputs"],
                "output_cols": params["output_cols"],
                "Y_min": params["Y_min"],
                "Y_max": params["Y_max"],
                "val_loss": best_val_loss,
            }, checkpoint)
        else:
            epochs_no_improve += 1

        # ── early stopping ─────────────────────────────────────────────────────
        if epochs_no_improve >= params["patience"]:
            logger.info(f"Early stop at epoch {epoch} (best val loss = {best_val_loss:.3e})")
            break

    logger.info(f"Training done. Best val loss = {best_val_loss:.3e}")
    logger.info(f"Checkpoint saved to {checkpoint}")


def main():
    """Main entry point for the flamelet neural-network trainer."""
    parser = argparse.ArgumentParser(
        description="Train a flamelet neural network from a TOML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="Path to TOML configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        config_dir = args.config.resolve().parent

        # Merge defaults with the [neural_net] section
        params = {**DEFAULTS, **config["neural_net"]}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {device}")

        # Resolve and check the solutions database
        solutions_file = resolve_solutions_file(config, config_dir)
        if not solutions_file.exists():
            raise ValueError(
                f"Solutions file not found: {solutions_file}\n"
                f"Run 'generate_table {args.config.name}' first to produce flamelet "
                f"solutions, or set [neural_net].solutions_file in the config."
            )

        # Load training data from raw flamelets
        inputs, Y, output_cols = load_training_data(
            solutions_file, params["inputs"], params["outputs"]
        )
        params["output_cols"] = output_cols
        logger.info(f"Loaded {len(Y):,} samples | {len(params['inputs'])} inputs "
                    f"({', '.join(params['inputs'])}) | {Y.shape[1]} outputs")

        # Build dataset / loaders
        dataset = FlametableDataset.from_raw(inputs, params["inputs"], Y)
        params["Y_min"] = dataset.Y_min
        params["Y_max"] = dataset.Y_max
        train_loader, val_loader = build_loaders(
            dataset, params["val_frac"], params["batch_size"], params["seed"]
        )

        # Build model
        model = FlametableNN(
            n_inputs=len(params["inputs"]),
            n_outputs=Y.shape[1],
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
        )
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Resolve checkpoint path
        checkpoint = Path(params.get("checkpoint", solutions_file.parent / "flamelet_nn.pt"))
        if not checkpoint.is_absolute():
            checkpoint = config_dir / checkpoint
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        train(model, train_loader, val_loader, params, checkpoint, device)

        logger.info("Flamelet NN training completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error training flamelet NN: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
