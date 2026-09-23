"""Aggregate the sweep's eval CSVs into sweep-level tables. Run from `scripts/`:

    python semisynthetic/aggregate.py --kind val_loss --split test

Reads   ../models/semisynthetic_v2/{axis}/{level}/rep{rep}/seed_{seed}/eval_v2_{split}_{kind}/
        {curve,effect,policy,factual}.csv        (written by semisynthetic/evaluate.py)
Writes  ../reports/semisynthetic_sweep_v2/
        {curve,effect,policy,factual}_raw.csv     one row per (axis, level, rep, kind, ...)
        checkpoint_kind.csv                        selected rule, pooled over available cells
        sweep_{axis}.csv                            one axis, mean +/- std over reps, fixed kind
        comparator_summary.csv                     every predictor, including KM effect errors
        factual_by_line.csv                        C-index/Brier by line at the reference cell

`best_constant` uses test truth and is an oracle benchmark for constant policies,
not a deployable fitted policy. `line_only` means eligibility is restricted by line
but not by patient-level propensity; predictions may still depend on patient history.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from CausalSurv.config import CHECKPOINT_KINDS

MODELS_DIR = Path("../models/semisynthetic_v2")
OUT_DIR = Path("../reports/semisynthetic_sweep_v2")
KINDS = CHECKPOINT_KINDS
CSVS = ["curve", "effect", "policy", "factual"]

EVAL_RE = re.compile(
    r"(?P<axis>[^/]+)/(?P<level>[^/]+)/rep(?P<rep>\d+)/seed_(?P<seed>\d+)/eval_(?:(?:v2_(?P<split>validation|test))_)?(?P<kind>\w+)$"
)


def _wavg(frame: pd.DataFrame, col: str, weight: str = "n") -> float:
    valid = frame[col].notna() & (frame[weight] > 0)
    f = frame.loc[valid]
    return (f[col] * f[weight]).sum() / f[weight].sum() if len(f) else float("nan")


def load_raw(
    models_dir: Path = MODELS_DIR,
    *,
    kind: str = "val_loss",
    split: str = "test",
    legacy_exploratory: bool = False,
) -> dict[str, pd.DataFrame]:
    """One concatenated, tagged frame per csv name, across every (axis, level, rep, kind)."""
    rows: dict[str, list[pd.DataFrame]] = {name: [] for name in CSVS}
    for eval_dir in sorted(models_dir.glob("*/*/rep*/seed_*/eval_*")):
        m = EVAL_RE.search(str(eval_dir))
        if not m or m["kind"] != kind:
            continue
        if legacy_exploratory:
            if m["split"] is not None:
                continue
        else:
            if m["split"] != split:
                continue
            metadata = json.loads((eval_dir / "metadata.json").read_text())
            if (
                metadata.get("evaluation_protocol") != 2
                or metadata.get("split") != split
                or metadata.get("checkpoint_kind") != kind
                or metadata.get("factual_estimator")
                != "exact_expected_and_uncensored_latent"
            ):
                raise ValueError(f"Incompatible evaluation metadata: {eval_dir}")
        tags = {
            "axis": m["axis"],
            "level": m["level"],
            "rep": int(m["rep"]),
            "seed": int(m["seed"]),
            "eval_kind": m["kind"],
            "evaluation_protocol": 1 if legacy_exploratory else 2,
            "split": "legacy_reused_holdout" if legacy_exploratory else split,
        }
        for name in CSVS:
            path = eval_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Incomplete evaluation: {path}")
            frame = pd.read_csv(path)
            frame["line"] = frame["line"].astype(str)
            for key, val in tags.items():
                frame.insert(0, key, val)
            rows[name].append(frame)
    if not rows["effect"]:
        raise ValueError(
            "No evaluations match the requested fixed kind/protocol/split. "
            "Retrain and evaluate with protocol v2, or explicitly use "
            "--legacy-exploratory for historical, non-confirmatory summaries."
        )
    return {name: pd.concat(parts, ignore_index=True) for name, parts in rows.items()}


def comparator_summary(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All comparators on identical support; seeds averaged within simulation replicate.

    `mean_pair_pehe` is an n-weighted mean of pair-specific root MSEs, not a
    pooled root MSE. The between-replicate SD is not a confidence interval.
    """
    keys = ["axis", "level", "rep", "seed", "eval_kind", "predictor"]
    records = []
    for group, eff in raw["effect"].query("line == 'all'").groupby(keys):
        tags = dict(zip(keys, group))

        def matching(name):
            frame = raw[name]
            for key, val in tags.items():
                frame = frame[frame[key] == val]
            return frame

        pol = matching("policy").query("line == 'all' and policy == 'line_only'")
        fac = matching("factual")
        curve = matching("curve").query("line == 'all' and kind == 'counterfactual'")
        records.append(
            {
                **tags,
                "mean_pair_pehe": _wavg(eff, "pehe"),
                "ate_bias_abs": _wavg(eff.assign(b=eff["ate_bias"].abs()), "b"),
                "counterfactual_rmst_mae": _wavg(curve, "rmst_mae"),
                "regret": _wavg(pol, "regret"),
                "hit_rate": _wavg(pol, "hit_rate"),
                "c_index": _wavg(fac, "c_index"),
                "brier": _wavg(fac, "brier"),
            }
        )
    runs = pd.DataFrame(records).drop(columns="seed")
    rep_keys = ["axis", "level", "eval_kind", "predictor", "rep"]
    reps = runs.groupby(rep_keys).mean(numeric_only=True).reset_index()
    metrics = [c for c in reps if c not in rep_keys]
    cells = reps.groupby(rep_keys[:-1])
    result = cells[metrics].agg(["mean", "std"])
    result.columns = ["_".join(c) for c in result.columns]
    result["n_reps"] = cells["rep"].nunique()
    return result.reset_index()


def checkpoint_kind_table(
    effect: pd.DataFrame, policy: pd.DataFrame, factual: pd.DataFrame
) -> pd.DataFrame:
    """Pooled over every cell and rep: does the checkpoint kind change counterfactual quality?"""
    eff = effect[(effect["predictor"] == "dynasurv") & (effect["line"] == "all")]
    pol = policy[
        (policy["predictor"] == "dynasurv")
        & (policy["policy"] == "line_only")
        & (policy["line"] == "all")
    ]
    fac = factual[(factual["predictor"] == "dynasurv")]
    out = []
    for kind in KINDS:
        e, p, f = (
            eff[eff["eval_kind"] == kind],
            pol[pol["eval_kind"] == kind],
            fac[fac["eval_kind"] == kind],
        )
        if not len(e):
            continue
        out.append(
            {
                "kind": kind,
                "n_runs": p["axis"].count(),
                "pehe": _wavg(e, "pehe"),
                "ate_bias_abs": _wavg(
                    e.assign(ate_bias_abs=e["ate_bias"].abs()), "ate_bias_abs"
                ),
                "sign_agree": _wavg(e, "sign_agree"),
                "regret": _wavg(p, "regret"),
                "hit_rate": _wavg(p, "hit_rate"),
                "c_index": _wavg(f, "c_index"),
                "brier": _wavg(f, "brier"),
            }
        )
    return pd.DataFrame(out)


def sweep_table(
    effect: pd.DataFrame,
    policy: pd.DataFrame,
    factual: pd.DataFrame,
    axis: str,
    kind: str,
) -> pd.DataFrame:
    """One axis, one checkpoint kind: mean +/- std over the 3 replicates, dynasurv vs references."""
    eff = effect[
        (effect["axis"] == axis)
        & (effect["eval_kind"] == kind)
        & (effect["line"] == "all")
    ]
    pol = policy[
        (policy["axis"] == axis)
        & (policy["eval_kind"] == kind)
        & (policy["line"] == "all")
    ]
    fac = factual[(factual["axis"] == axis) & (factual["eval_kind"] == kind)]

    rows = []
    for level, e_lvl in eff[eff["predictor"] == "dynasurv"].groupby("level"):
        by_rep_pehe = e_lvl.groupby("rep").apply(
            lambda g: _wavg(g, "pehe"), include_groups=False
        )
        by_rep_bias = e_lvl.groupby("rep").apply(
            lambda g: _wavg(g.assign(b=g["ate_bias"].abs()), "b"), include_groups=False
        )
        p_lvl = pol[
            (pol["level"] == level)
            & (pol["predictor"] == "dynasurv")
            & (pol["policy"] == "line_only")
        ]
        p_ref_best = pol[
            (pol["level"] == level)
            & (pol["predictor"] == "reference")
            & (pol["policy"] == "best_constant")
        ]
        p_ref_rand = pol[
            (pol["level"] == level)
            & (pol["predictor"] == "reference")
            & (pol["policy"] == "random")
        ]
        p_oracle = pol[
            (pol["level"] == level)
            & (pol["predictor"] == "oracle")
            & (pol["policy"] == "line_only")
        ]
        by_rep_regret = p_lvl.groupby("rep")["regret"].mean()
        by_rep_hit = p_lvl.groupby("rep")["hit_rate"].mean()
        f_lvl = fac[(fac["level"] == level) & (fac["predictor"] == "dynasurv")]
        by_rep_c = f_lvl.groupby("rep").apply(
            lambda g: _wavg(g, "c_index"), include_groups=False
        )
        f_naive = fac[(fac["level"] == level) & (fac["predictor"] == "naive_km")]
        f_oracle = fac[(fac["level"] == level) & (fac["predictor"] == "oracle")]
        rows.append(
            {
                "axis": axis,
                "level": level,
                "n_reps": e_lvl["rep"].nunique(),
                "pehe_mean": by_rep_pehe.mean(),
                "pehe_std": by_rep_pehe.std(),
                "ate_bias_abs_mean": by_rep_bias.mean(),
                "ate_bias_abs_std": by_rep_bias.std(),
                "regret_mean": by_rep_regret.mean(),
                "regret_std": by_rep_regret.std(),
                "hit_rate_mean": by_rep_hit.mean(),
                "hit_rate_std": by_rep_hit.std(),
                "c_index_mean": by_rep_c.mean(),
                "c_index_std": by_rep_c.std(),
                "c_index_naive_km": f_naive.groupby("rep")
                .apply(lambda g: _wavg(g, "c_index"), include_groups=False)
                .mean(),
                "c_index_oracle": f_oracle.groupby("rep")
                .apply(lambda g: _wavg(g, "c_index"), include_groups=False)
                .mean(),
                "regret_best_constant": p_ref_best.groupby("rep")["regret"]
                .mean()
                .mean(),
                "regret_random": p_ref_rand.groupby("rep")["regret"].mean().mean(),
                "regret_oracle": p_oracle.groupby("rep")["regret"].mean().mean(),
            }
        )
    return pd.DataFrame(rows)


def factual_by_line(
    factual: pd.DataFrame, axis: str, level: str, kind: str
) -> pd.DataFrame:
    sub = factual[
        (factual["axis"] == axis)
        & (factual["level"] == level)
        & (factual["eval_kind"] == kind)
    ]
    return (
        sub.groupby(["predictor", "line"])[["c_index", "brier"]]
        .agg(["mean", "std"])
        .round(3)
        .reset_index()
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default="val_loss",
        help="Prespecified/development-selected rule; never optimized on these tables.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--legacy-exploratory",
        action="store_true",
        help="Summarize historical reused-holdout tables, explicitly not corrected validation.",
    )
    args = parser.parse_args()
    raw = load_raw(
        args.models_dir,
        kind=args.kind,
        split=args.split,
        legacy_exploratory=args.legacy_exploratory,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary_protocol.json").write_text(
        json.dumps(
            {
                "checkpoint_kind": args.kind,
                "selection": "fixed_input_not_test_optimized",
                "evaluation_protocol": 1 if args.legacy_exploratory else 2,
                "split": "legacy_reused_holdout"
                if args.legacy_exploratory
                else args.split,
                "evaluation_role": "exploratory"
                if args.legacy_exploratory
                else args.split,
                "test_not_used_for_selection_by_aggregator": not args.legacy_exploratory,
                "factual_estimator": "legacy_ipcw"
                if args.legacy_exploratory
                else "exact_expected_and_uncensored_latent",
            },
            indent=2,
        )
    )
    for name, frame in raw.items():
        frame.to_csv(args.out / f"{name}_raw.csv", index=False)

    comparison = comparator_summary(raw)
    comparison.to_csv(args.out / "comparator_summary.csv", index=False)
    print(
        "\nALL COMPARATORS (same support; mean pair PEHE is not pooled PEHE)\n",
        comparison.round(3).to_string(index=False),
    )

    ck = checkpoint_kind_table(raw["effect"], raw["policy"], raw["factual"])
    ck.to_csv(args.out / "checkpoint_kind.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(
        "\nCHECKPOINT KIND (pooled over all cells and reps)\n",
        ck.round(3).to_string(index=False),
    )

    selected_kind = args.kind
    print(
        f"\nFixed checkpoint rule: {selected_kind}; no selection on reported oracle regret."
    )
    if args.legacy_exploratory:
        print(
            "WARNING: legacy reused-holdout results are exploratory, not corrected validation."
        )

    for axis in ["gamma", "strength", "heterogeneity"]:
        table = sweep_table(
            raw["effect"], raw["policy"], raw["factual"], axis, selected_kind
        )
        table.round(3).to_csv(args.out / f"sweep_{axis}.csv", index=False)
        print(
            f"\nSWEEP: {axis} [{selected_kind}]\n",
            table.round(3).to_string(index=False),
        )

    fbl = factual_by_line(raw["factual"], "gamma", "1.0", selected_kind)
    fbl.to_csv(args.out / "factual_by_line.csv", index=False)
    print(
        f"\nFACTUAL BY LINE, reference cell (gamma=1.0) [{selected_kind}]\n",
        fbl.to_string(index=False),
    )

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    cli()
