# plot_projection.py
"""
Script to plot 3D manifold projection from requested variables.
Usage:
    python plot_flamelets.py <solutions_file> [-o <output_dir>] [-v]
"""
import argparse
import logging
from pathlib import Path

from fpvgen.flamelet_table_generator import FlameletTableGenerator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot flamelet solutions from HDF5 file")
    parser.add_argument("solutions_file", type=Path, help="Path to HDF5 solutions file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for plots (default: same as solutions file)",
    )
    parser.add_argument(
        "--species",
        "-s",
        type=str,
        help="Comma separated 3 species in the manifold, no spaces (default: O2,H2,H2O)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Load solutions
        logger.info(f"Loading solutions from {args.solutions_file}")
        generator = FlameletTableGenerator.load_solutions(args.solutions_file)

        # Set output directory
        output_dir = args.output or args.solutions_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Obtain species list
        species_str = args.species

        if species_str is None:
            species_str = "O2,H2,H2O"

        species_list = species_str.split(",")
        vars = [s+"-massfraction" for s in species_list]

        # Generate manifold plot
        logger.info("Generating manifold projection")
        generator.assemble_data_table()
        generator.plot_3d_projection(vars=vars, output_file=output_dir / f"manifold_{species_list[0]}_{species_list[1]}_{species_list[2]}.png")

        logger.info("Plot generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    main()
