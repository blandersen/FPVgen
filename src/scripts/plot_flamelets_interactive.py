# plot_flamelets_interactive.py
"""
Script to interactively inspect flamelet solutions and select misbehaving ones to omit.

Opens a matplotlib GUI to plot the flamelets (scatter of per-flamelet scalars, or profile
curves), click to toggle which flamelets are omitted, and writes a filtered solutions file
containing only the kept flamelets.

Usage:
    python plot_flamelets_interactive.py <solutions_file> [-o <output_dir>] [-v]

Note: requires an interactive matplotlib backend (e.g. set MPLBACKEND=TkAgg or Qt5Agg).
"""
import argparse
import logging
from pathlib import Path

from fpvgen.flamelet_table_generator import FlameletTableGenerator


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactively inspect flamelet solutions and omit misbehaving ones"
    )
    parser.add_argument("solutions_file", type=Path, help="Path to HDF5 solutions file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for the filtered solutions file (default: same as solutions file)",
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

        # Launch the interactive GUI
        result = generator.plot_flamelets_interactive(output_dir=output_dir)

        logger.info(f"Kept {len(result['kept'])} flamelets: {result['kept']}")
        logger.info(f"Omitted {len(result['omitted'])} flamelets: {result['omitted']}")
        return 0

    except Exception as e:
        logger.error(f"Error during interactive plotting: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    main()
