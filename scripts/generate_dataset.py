from __future__ import annotations

import argparse
from pathlib import Path

from eiga.data.generator import DatasetGenerationConfig, SupervisedDatasetGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scenario-balanced EIGA supervised datasets."
    )
    parser.add_argument("--output", type=Path, default=Path("data/generated/default"))
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--validation", type=int, default=50)
    parser.add_argument("--test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--diagnostic-probability", type=float, default=0.30)
    parser.add_argument("--max-prefixes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetGenerationConfig(
        train_episodes_per_scenario=args.train,
        validation_episodes_per_scenario=args.validation,
        test_episodes_per_scenario=args.test,
        base_seed=args.seed,
        diagnostic_probability=args.diagnostic_probability,
        max_prefixes_per_episode=args.max_prefixes,
    )
    paths = SupervisedDatasetGenerator(generation_config=config).write(args.output)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
