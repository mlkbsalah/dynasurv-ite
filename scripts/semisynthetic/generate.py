"""Generate one semi-synthetic replicate. Run from `scripts/`:

    python semisynthetic/generate.py --out ../data/semisynthetic/gamma/1.0/rep0 \
        --replicate 0 --gamma 1.0
"""

import argparse

from CausalSurv.semisynthetic.config import DGPConfig
from CausalSurv.semisynthetic.generate import generate, with_knobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="../configs/semisynthetic/dgp.toml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--heterogeneity", type=float)
    args = parser.parse_args()

    cfg = with_knobs(
        DGPConfig.from_file(args.config),
        gamma=args.gamma,
        strength=args.strength,
        heterogeneity=args.heterogeneity,
    )
    manifest = generate(cfg, args.out, args.replicate)
    print(
        f"wrote {args.out}: {manifest['n_samples']} samples, "
        f"{manifest['n_expanded_rows']} rows"
    )


if __name__ == "__main__":
    main()
