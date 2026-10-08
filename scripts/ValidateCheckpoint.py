"""Save factual development-validation figures and tables for one checkpoint.

Run from any directory:
    python3 scripts/ValidateCheckpoint.py /path/to/run/checkpoints/model.ckpt

Outputs are written beside the checkpoint's ``checkpoints/`` directory. The
reserved temporal test partition is never scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import lightning as L
import matplotlib
import numpy as np
import tomllib
import torch
from sksurv.nonparametric import kaplan_meier_estimator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from CausalSurv.config import DataConfig, EvalConfig  # noqa: E402
from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV  # noqa: E402
from CausalSurv.evaluation.calibration import compute_calibration  # noqa: E402
from CausalSurv.model.checkpoint_compat import load_dynasurv_checkpoint  # noqa: E402
from CausalSurv.recommendation import (  # noqa: E402
    NO_SUPPORTED_ARM,
    TreatmentRecommender,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/config.toml"
CALIBRATION_TIMES = (6.0, 12.0, 18.0, 24.0)


def save_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Checkpoint must be inside a run's checkpoints/ directory")
    run_dir = checkpoint.parent.parent
    match = re.search(r"_seed_(\d+)$", run_dir.name)
    if match is None:
        raise ValueError(f"Cannot infer the split seed from {run_dir.name!r}")

    model = load_dynasurv_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    with CONFIG_PATH.open("rb") as stream:
        run_config = tomllib.load(stream)
    data_config = DataConfig.from_dict(run_config["data"])
    eval_config = EvalConfig.from_dict(run_config["eval"])
    kwargs = data_config.to_dict()
    kwargs["data_dir"] = str((CONFIG_PATH.parent / data_config.data_dir).resolve())
    kwargs["excluded_treatment_arms"] = list(data_config.excluded_treatment_arms)
    kwargs["excluded_x_columns"] = list(data_config.excluded_x_columns)
    kwargs["evaluation_horizon_times"] = list(eval_config.horizon_times)
    kwargs["n_intervals"] = model.output_length
    kwargs["batch_size"] = 256
    dm = ESMEOnlineDataModuleCV(
        **kwargs,
        split_seed=int(match.group(1)),
        final_training=True,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup("fit")
    manifest = json.loads(json.dumps(dm.data_manifest))
    if (
        model.data_manifest is None
        or json.loads(json.dumps(model.data_manifest)) != manifest
    ):
        raise ValueError(
            "Checkpoint data manifest differs from the current cohort/split"
        )
    manifest_path = run_dir / "data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run manifest differs from the current cohort/split")
    dims = dm.get_data_dimensions()
    for key, expected in (
        ("x_input_dim", model.x_input_dim),
        ("x_static_dim", model.x_static_dim),
        ("p_input_dim", model.p_input_dim),
        ("p_static_dim", model.p_static_dim),
        ("output_dim", model.output_length),
    ):
        if dims[key] != expected:
            raise ValueError(
                f"{key}: rebuilt data has {dims[key]}, checkpoint has {expected}"
            )

    model.fit_censoring_estimator(dm.train_dataloader())
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    metrics = trainer.validate(model, dataloaders=dm.val_dataloader(), verbose=False)[0]
    metric_rows = [
        {"metric": key, "value": float(value)}
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float, torch.Tensor)) and math.isfinite(float(value))
    ]
    save_rows(run_dir / "validation_metrics.csv", metric_rows, ["metric", "value"])

    with torch.inference_mode():
        XPd, static, _, treatment, time, event, mask, _ = next(
            iter(dm.val_dataloader())
        )
        survival = (
            model.predict_discrete_survival(
                XPd, static, gather=True, factual_idx=treatment
            )
            .cpu()
            .numpy()
        )
    time = time.squeeze(-1).cpu().numpy()
    event = event.squeeze(-1).cpu().numpy().astype(bool)
    mask = mask.cpu().numpy().astype(bool)
    treatment = treatment.cpu().numpy()
    bounds = model.interval_bounds.cpu().numpy()
    n_lines = mask.shape[1]

    # The notebook's treatment distribution and follow-up diagnostics.
    count_rows = []
    for line in range(n_lines):
        values, counts = np.unique(treatment[mask[:, line], line], return_counts=True)
        count_rows.extend(
            {
                "line": line + 1,
                "treatment": dm.treatment_dict[int(value)],
                "count": int(count),
            }
            for value, count in zip(values, counts)
        )
    save_rows(
        run_dir / "validation_treatment_counts.csv",
        count_rows,
        ["line", "treatment", "count"],
    )
    fig, axes = plt.subplots(1, n_lines, figsize=(4 * n_lines, 3), squeeze=False)
    for line, ax in enumerate(axes[0]):
        observed = time[mask[:, line], line]
        ax.hist(observed, bins=30, color="steelblue")
        ax.set(
            title=f"Line {line + 1} (n={len(observed)})",
            xlabel="Follow-up (months)",
            ylabel="Patients",
        )
    save_figure(fig, run_dir / "validation_followup.png")

    # Factual average survival versus Kaplan-Meier, using the same observed
    # patients at each treatment line. These are descriptive marginal curves.
    fig, axes = plt.subplots(1, n_lines, figsize=(4.5 * n_lines, 3.5), squeeze=False)
    for line, ax in enumerate(axes[0]):
        valid = mask[:, line]
        if not valid.any():
            continue
        km_t, km_s = kaplan_meier_estimator(event[valid, line], time[valid, line])
        ax.step(
            bounds,
            survival[valid, line].mean(axis=0),
            where="post",
            label="Mean prediction",
        )
        ax.step(km_t, km_s, where="post", linestyle="--", label="Kaplan-Meier")
        ax.set(
            title=f"Line {line + 1}",
            xlabel="Months",
            ylabel="Survival",
            xlim=(0, eval_config.horizon_times[line]),
            ylim=(0, 1.02),
        )
        ax.legend()
    save_figure(fig, run_dir / "validation_survival_vs_km.png")

    # Training-cohort censoring reference, as in the model's epoch metrics.
    brier_rows = []
    fig, axes = plt.subplots(1, n_lines, figsize=(4.5 * n_lines, 3.5), squeeze=False)
    for line, ax in enumerate(axes[0]):
        valid = mask[:, line]
        if not valid.any():
            continue
        _, curve, _ = model.eval_brier_score_ipcw(
            train_events=torch.as_tensor(model.train_events[line], dtype=torch.bool),
            train_times=torch.as_tensor(model.train_times[line], dtype=torch.float32),
            test_events=torch.as_tensor(event[valid, line]),
            test_times=torch.as_tensor(time[valid, line], dtype=torch.float32),
            discrete_survival=torch.as_tensor(survival[valid, line]),
            tmax=eval_config.horizon_times[line],
        )
        grid = np.linspace(0, eval_config.horizon_times[line], len(curve))
        values = curve.cpu().numpy()
        brier_rows.extend(
            {"line": line + 1, "month": float(month), "brier_score": float(value)}
            for month, value in zip(grid, values)
        )
        ax.plot(grid, values)
        ax.set(title=f"Line {line + 1}", xlabel="Months", ylabel="IPCW Brier score")
    save_rows(
        run_dir / "validation_brier_curve.csv",
        brier_rows,
        ["line", "month", "brier_score"],
    )
    save_figure(fig, run_dir / "validation_brier_curve.png")

    # Notebook-style binned calibration at fixed landmarks. Empty bins are
    # skipped by compute_calibration; ECE is undefined if no bin survives.
    calibration_rows = []
    fig, axes = plt.subplots(1, n_lines, figsize=(4.5 * n_lines, 3.5), squeeze=False)
    for line, ax in enumerate(axes[0]):
        valid = mask[:, line]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        if not valid.any():
            continue
        for month in CALIBRATION_TIMES:
            if month > eval_config.horizon_times[line]:
                continue
            predicted = (
                model.eval_factual_survival(
                    torch.as_tensor(survival[valid, line]),
                    torch.tensor([month], dtype=torch.float32),
                )
                .cpu()
                .numpy()
                .ravel()
            )
            pred_bin, obs_bin, counts = compute_calibration(
                torch.as_tensor(predicted),
                time[valid, line],
                event[valid, line],
                month,
                n_bins=10,
            )
            if len(counts) == 0:
                continue
            ece = float(np.average(np.abs(obs_bin - pred_bin), weights=counts))
            for pred, obs, count in zip(pred_bin, obs_bin, counts):
                calibration_rows.append(
                    {
                        "line": line + 1,
                        "month": month,
                        "predicted": float(pred),
                        "observed_km": float(obs),
                        "count": int(count),
                        "ece": ece,
                    }
                )
            ax.plot(pred_bin, obs_bin, ".-", label=f"{month:g} mo")
        ax.set(
            title=f"Line {line + 1}",
            xlabel="Predicted survival",
            ylabel="Observed KM survival",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    save_rows(
        run_dir / "validation_calibration.csv",
        calibration_rows,
        ["line", "month", "predicted", "observed_km", "count", "ece"],
    )
    save_figure(fig, run_dir / "validation_calibration.png")

    # Descriptive support-masked recommendation mix from the notebook. The
    # observed-versus-recommended comparison is not a treatment-effect estimate.
    recommender = TreatmentRecommender.from_model(model)
    with torch.inference_mode():
        recommended, _ = recommender.recommend(
            XPd,
            static,
            factual_idx=torch.as_tensor(treatment),
            horizon_times=list(eval_config.horizon_times),
        )
    recommended = recommended.cpu().numpy()
    mix_rows = []
    for line in range(n_lines):
        values, counts = np.unique(recommended[mask[:, line], line], return_counts=True)
        mix_rows.extend(
            {
                "line": line + 1,
                "recommendation": (
                    "no_supported_arm"
                    if value == NO_SUPPORTED_ARM
                    else dm.treatment_dict[int(value)]
                ),
                "count": int(count),
            }
            for value, count in zip(values, counts)
        )
    save_rows(
        run_dir / "validation_recommendation_mix.csv",
        mix_rows,
        ["line", "recommendation", "count"],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = sorted({row["recommendation"] for row in mix_rows})
    bottoms = np.zeros(n_lines)
    for label in labels:
        heights = np.array(
            [
                next(
                    (
                        row["count"]
                        for row in mix_rows
                        if row["line"] == line + 1 and row["recommendation"] == label
                    ),
                    0,
                )
                for line in range(n_lines)
            ]
        )
        ax.bar(np.arange(1, n_lines + 1), heights, bottom=bottoms, label=label)
        bottoms += heights
    ax.set(
        xlabel="Treatment line",
        ylabel="Validation patients",
        title="Supported recommendation mix",
    )
    ax.set_xticks(range(1, n_lines + 1))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save_figure(fig, run_dir / "validation_recommendation_mix.png")

    (run_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "split": "development_validation",
                "patients": len(dm.val_dataset),
                "note": "Factual metrics and descriptive recommendation mix; no temporal test or causal policy effect.",
                "figures": [
                    path.name for path in sorted(run_dir.glob("validation_*.png"))
                ],
                "tables": [
                    path.name for path in sorted(run_dir.glob("validation_*.csv"))
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Validation outputs: {run_dir}")


if __name__ == "__main__":
    run()
