# run_single_experiment.py
"""
Runs exactly one experiment and exits.
All arguments passed via CLI — no shared state with any other process.
"""

import argparse
import numpy as np

from helpers import save_results
from model import FullyConnectedRBM
from sampler import ClassicalSampler, DimodSampler, VeloxSampler
from encoder import Trainer
from ising import TransverseFieldIsing1D, TransverseFieldIsing2D
from argparse import Namespace


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--sampler", type=str, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", type=str, default="results/")
    p.add_argument("--model", choices=["1d", "2d"], default="1d")
    (
        p.add_argument(
            "--n-hidden",
            type=int,
            default=None,
            help="Number of hidden units. Defaults to size (1D) or size (2D linear dim).",
        ),
    )
    (p.add_argument("--iterations", type=int, default=300),)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    n_visible = args.size if args.model == "1d" else args.size**2
    # 1. Instantiate Ising model
    if args.model == "1d":
        ising = TransverseFieldIsing1D(args.size)
    elif args.model == "2d":
        ising = TransverseFieldIsing2D(args.size)
    n_hidden = args.n_hidden if args.n_hidden is not None else args.size
    rbm = FullyConnectedRBM(n_visible, n_hidden)

    if args.sampler == "custom":
        sampler = ClassicalSampler(method=args.method)
    elif args.sampler == "dimod":
        sampler = DimodSampler(method=args.method)
    elif args.sampler == "velox":
        sampler = VeloxSampler(method=args.method)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    trainer_config = {
        "learning_rate": args.lr,
        "n_iterations": args.iterations,
        "n_samples": 1000,
        "regularization": 1e-3,
        "stop_at_convergence": False,
    }

    ns_args = Namespace(
        model=args.model,
        size=args.size,
        h=0.5,
        rbm="full",
        n_hidden=n_hidden,
        sampler=args.sampler,
        sampling_method=args.method,
        iterations=args.iterations,
        learning_rate=args.lr,
        regularization=1e-3,
        n_samples=1000,
        output_dir=args.output_dir,
        seed=args.seed,
        visualize=False,
    )

    trainer = Trainer(rbm, ising, sampler, trainer_config)
    history = trainer.train()
    save_results(ns_args, history, ising)


if __name__ == "__main__":
    main()
