# This file is intentionally empty.
# CEM comparison plotting is in scripts/plot_cem_comparison.py
# Run: python scripts/plot_cem_comparison.py [--size N] [--h H] [--model 1d|2d]
#
# To generate data for comparison, run main.py twice with the same args,
# once without --cem and once with --cem:
#   python src/main.py --model 1d --size 8 --h 0.5
#   python src/main.py --model 1d --size 8 --h 0.5 --cem
