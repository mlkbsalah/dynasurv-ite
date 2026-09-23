"""Recoverably quarantine pre-protocol-v2 artifacts; dry-run unless --apply.

Only trusted, local project checkpoints are loaded (PyTorch pickle format).
Modern/ambiguous runs are protected, even when sharing a tree with old runs.
No deletion, compression, overwriting, or changes to other worktrees occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_checkpoint(path: Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # A manifest or static-state projections indicate newer work: fail closed,
    # including incomplete migrations, not just a specific version number.
    return (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("state_dict"), dict)
        and not checkpoint.get("data_manifest")
        and not checkpoint.get("hyper_parameters", {}).get("use_static_features")
        and not any(
            key.startswith(("init_h.", "init_c.", "init_p."))
            for key in checkpoint["state_dict"]
        )
    )


def legacy_study(path: Path) -> bool:
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(str(path) + suffix).exists():
            raise RuntimeError(f"Database may be open; refusing to move {path}")
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"Integrity check failed: {path}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"studies", "trials", "study_user_attributes"}.issubset(tables):
            return False
        return not connection.execute(
            "SELECT 1 FROM study_user_attributes WHERE key IN "
            "('data_protocol', 'evaluation_protocol') LIMIT 1"
        ).fetchone()


def inventory(root: Path) -> tuple[list[Path], dict]:
    targets, protected = [], []
    checkpoint_count = run_count = 0
    models = root / "models"
    for tree in sorted(models.iterdir()) if models.exists() else []:
        if not tree.is_dir() or tree.is_symlink():
            continue
        checkpoints = sorted(tree.rglob("*.ckpt"))
        if not checkpoints:
            continue
        # Log symlinks are preserved as links, never followed. A linked model
        # or manifest is ambiguous and must not be loaded or classified.
        if any(p.is_symlink() for p in checkpoints) or any(
            p.is_symlink() for p in tree.rglob("data_manifest.json")
        ):
            protected.append(str(tree.relative_to(root)))
            continue
        modern_manifest = any(
            json.loads(p.read_text()).get("protocol_version", 0) >= 2
            for p in tree.rglob("data_manifest.json")
        )
        if modern_manifest or not all(legacy_checkpoint(p) for p in checkpoints):
            protected.append(str(tree.relative_to(root)))
            # A new run may coexist with old runs under the same subtype.
            # Quarantine only complete legacy run directories, not their parent.
            for run in sorted(
                {p.parent.parent for p in checkpoints if p.parent.name == "checkpoints"}
            ):
                run_checkpoints = sorted(run.rglob("*.ckpt"))
                if list(run.rglob("data_manifest.json")):
                    continue
                if all(
                    not p.is_symlink() and legacy_checkpoint(p) for p in run_checkpoints
                ):
                    targets.append(run)
                    checkpoint_count += len(run_checkpoints)
                    run_count += 1
            continue
        targets.append(tree)
        checkpoint_count += len(checkpoints)
        run_count += len({p.parent for p in checkpoints})

    study_count = 0
    for name in ("optuna_study.db", "optuna_synth.db"):
        path = root / "scripts" / "hyperopt" / name
        if path.exists() and legacy_study(path):
            targets.append(path)
            study_count += 1
            log = path.with_suffix(".log")
            if log.exists():
                targets.append(log)
            # These exact files are the winners associated with the old studies.
            config = (
                root
                / "configs"
                / (
                    "best_config.json"
                    if name == "optuna_study.db"
                    else "optuna_best_synthetic.toml"
                )
            )
            if config.exists():
                targets.append(config)
        elif path.exists():
            protected.append(str(path.relative_to(root)))
    # Preserve aborted, log-only runs from before this protocol migration too.
    # Both logged start times and all artifact mtimes must predate the cutoff;
    # a resumed or new run is never moved based only on its directory name.
    cutoff = datetime(2026, 9, 23, tzinfo=timezone.utc)
    log_only_runs = 0
    for run in sorted(models.rglob("*_seed_*")) if models.exists() else []:
        if (
            not run.is_dir()
            or run.is_symlink()
            or any(run.is_relative_to(p) for p in targets)
        ):
            continue
        metadata = list(run.rglob("wandb-metadata.json"))
        if (
            not metadata
            or list(run.rglob("*.ckpt"))
            or list(run.rglob("data_manifest.json"))
        ):
            continue
        files = [p for p in run.rglob("*") if p.is_file() or p.is_symlink()]
        if all(
            not p.is_symlink()
            and datetime.fromisoformat(
                json.loads(p.read_text())["startedAt"].replace("Z", "+00:00")
            )
            < cutoff
            for p in metadata
        ) and all(p.lstat().st_mtime < cutoff.timestamp() for p in files):
            targets.append(run)
            log_only_runs += 1
    return targets, dict(
        checkpoints=checkpoint_count,
        runs=run_count,
        log_only_runs=log_only_runs,
        study_databases=study_count,
        protected=protected,
    )


def archive(
    root: Path, destination: Path, *, apply: bool = False, append: bool = False
) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if destination.parent != root / "archives":
        raise ValueError(
            "Archive must be a named directory directly inside PROJECT/archives"
        )
    if destination.exists() and not append:
        raise FileExistsError(f"Will not overwrite an archive: {destination}")
    previous = None
    if destination.exists():
        previous = json.loads((destination / "manifest.json").read_text())
        if not previous.get("verified"):
            raise ValueError("Cannot append to an incomplete/unverified archive")
    targets, summary = inventory(root)
    entries = []
    for source in targets:
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe artifact path: {source}")
        files = (
            sorted(p for p in source.rglob("*") if p.is_symlink() or p.is_file())
            if source.is_dir()
            else [source]
        )
        for path in files:
            if path.is_symlink():
                target = os.readlink(path)
                entries.append(
                    dict(
                        path=str(path.relative_to(root)),
                        type="symlink",
                        target=target,
                        size=len(os.fsencode(target)),
                        sha256=hashlib.sha256(os.fsencode(target)).hexdigest(),
                    )
                )
            else:
                entries.append(
                    dict(
                        path=str(path.relative_to(root)),
                        type="file",
                        size=path.stat().st_size,
                        sha256=sha256(path),
                    )
                )
    summary.update(
        files=len(entries),
        bytes=sum(e["size"] for e in entries),
        targets=[str(p.relative_to(root)) for p in targets],
    )
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        reason="Pre-v2 data/static-feature and pre-v3 HPO protocols; historical only",
        summary=summary,
        files=entries,
        moved=[],
        verified=False,
    )
    if previous:
        for entry in entries:
            if (destination / "artifacts" / entry["path"]).exists():
                raise FileExistsError(entry["path"])
        manifest["created_utc"] = previous["created_utc"]
        manifest["files"] = previous["files"] + entries
        manifest["moved"] = previous["moved"].copy()
        manifest["summary"] = {
            key: previous["summary"].get(key, [] if isinstance(value, list) else 0)
            + value
            if key != "protected"
            else value
            for key, value in summary.items()
        }
    if apply:
        if not targets:
            raise ValueError("No verified legacy artifacts to archive")
        destination.mkdir(parents=True, exist_ok=append)
        manifest_path = destination / "manifest.json"

        def save_manifest():
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        save_manifest()
        for source in targets:
            relative = source.relative_to(root)
            target = destination / "artifacts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
            manifest["moved"].append(str(relative))
            save_manifest()
        for entry in entries:
            target = destination / "artifacts" / entry["path"]
            if entry["type"] == "symlink":
                valid = target.is_symlink() and os.readlink(target) == entry["target"]
            else:
                valid = (
                    target.stat().st_size == entry["size"]
                    and sha256(target) == entry["sha256"]
                )
            if not valid:
                raise RuntimeError(f"Archive verification failed: {target}")
        manifest["verified"] = True
        save_manifest()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--name", default="legacy_pre_v2_2026-09-23")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to a verified archive; never overwrite artifacts",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            archive(
                args.root,
                args.root / "archives" / args.name,
                apply=args.apply,
                append=args.append,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
