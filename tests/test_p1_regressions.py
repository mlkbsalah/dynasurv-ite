"""Regression coverage for audit M1--M3, R1/R2/R4, and S1--S3."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torchsurv.stats.ipcw import get_ipcw

from CausalSurv.config import ArchConfig, EvalConfig, TrainingConfig
from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.evaluation.propensity_overlap import PropensityOverlapModel
from CausalSurv.model.checkpoint_compat import load_dynasurv_checkpoint
from CausalSurv.model.dynasurv_causal_online import DynaSurvCausalOnline
from CausalSurv.recommendation.ensemble import (
    IncompatibleMemberError,
    check_members,
    load_member,
)
from CausalSurv.recommendation.recommender import TreatmentRecommender
from CausalSurv.semisynthetic.datamodule import SemiSyntheticDataModule
from CausalSurv.semisynthetic.evaluate import build_truth, evaluate, factual_metrics
from CausalSurv.semisynthetic.predictors import Prediction, oracle_prediction


def small_model(**kwargs):
    torch.manual_seed(11)
    return DynaSurvCausalOnline(
        x_input_dim=2,
        x_static_dim=2,
        p_input_dim=3,
        p_static_dim=2,
        n_treatments=3,
        output_length=6,
        interval_bounds=torch.arange(7).float(),
        n_lines=2,
        arch=ArchConfig(8, 1, 4, 4, (8,), (8,), (8,), 0.0, 0.0, 0.0, False),
        training=TrainingConfig(0.001, 0.0, 10, 0.9),
        evaluation=EvalConfig(
            (3.0, 3.0), integration_step=3, calibration_times=(1.0, 2.0)
        ),
        **kwargs,
    ).eval()


def model_inputs():
    xpd = torch.randn(8, 2, 6)
    xpd[:, :, -1] = 0.5
    static = (torch.randn(8, 2), torch.randn(8, 2))
    treatments = torch.zeros(8, 2, dtype=torch.long)
    return xpd, static, treatments


def test_statics_affect_predictions_and_receive_gradient_without_current_arm_leak():
    model = small_model()
    xpd, static, treatments = model_inputs()
    base = model(xpd, static, treatments)[0]
    for which in (0, 1):
        changed = list(static)
        changed[which] = changed[which] + 2
        assert not torch.allclose(base, model(xpd, tuple(changed), treatments)[0])
    base.square().sum().backward()
    for projection in (model.init_h, model.init_c, model.init_p):
        assert projection.weight.grad.abs().sum() > 0
    changed_treatment = treatments.clone()
    changed_treatment[:, -1] = 2
    assert torch.equal(base, model(xpd, static, changed_treatment)[0])


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_roundtrip_preserves_static_mode_and_manifest(tmp_path, legacy):
    model = small_model(use_static_features=not legacy)
    inputs = model_inputs()
    model.data_manifest = (
        None if legacy else {"protocol_version": 2, "splits": {"test": [2]}}
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
    }
    model.on_save_checkpoint(checkpoint)
    if legacy:
        checkpoint["hyper_parameters"].pop("use_static_features")
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    restored = load_dynasurv_checkpoint(path).eval()
    assert restored.use_static_features is not legacy
    assert restored.data_manifest == model.data_manifest
    assert torch.equal(model(*inputs)[0], restored(*inputs)[0])


def test_trainer_cannot_overwrite_checkpoint_provenance_with_a_different_split():
    model = small_model()
    model.data_manifest = {"protocol_version": 2, "splits": {"test": [1]}}
    model._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            data_manifest={"protocol_version": 2, "splits": {"test": [2]}}
        )
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        model._setup_valid_treatments()
    assert model.data_manifest["splits"]["test"] == [1]


def frames(expanded=False):
    rows, statics = [], []
    for original in range(30):
        for endpoint in (1, 2) if expanded else (2,):
            pid = original * 10 + endpoint if expanded else original
            for line in range(1, endpoint + 1):
                row = dict(
                    usubjid=pid,
                    lineid=line,
                    line_start_date=pd.Timestamp(
                        2019 if original < 24 else 2021, 1, line
                    ),
                    X_marker=float(original + line),
                    X_time_between_onsets=float(line - 1),
                    T_treatment_category="A" if original % 2 else "B",
                    Y_onset_to_death=float(5 + original % 15),
                    Y_global_death_status=original % 3 != 0,
                )
                if expanded:
                    row.update(orig_usubjid=original, prefix_line=endpoint)
                rows.append(row)
            statics.append(
                dict(usubjid=pid, X_age=float(40 + original), T_prior=original % 2)
            )
    return pd.DataFrame(rows), pd.DataFrame(statics)


def data_module(dynamic=None, static=None, *, expanded=False, **overrides):
    if dynamic is None:
        dynamic, static = frames(expanded)
    cls = SemiSyntheticDataModule if expanded else ESMEOnlineDataModuleCV
    defaults = dict(
        data_dir="unused",
        subtype="HR+HER2-",
        n_lines=2,
        n_intervals=6,
        batch_size=8,
        split_seed=0,
        final_training=True,
        num_workers=0,
        temporal_split_year=2021,
        min_samples_per_treatment=1,
        propensity_cv_folds=2,
        excluded_treatment_arms=[],
    )
    dm = cls(**(defaults | overrides))
    dm._load_data = lambda: (dynamic.copy(), static.copy())
    dm.prepare_data()
    return dm


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("final_training", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_splits_are_disjoint_by_original_patient_and_cv_never_sees_test(
    expanded, final_training, temporal
):
    dm = data_module(
        expanded=expanded,
        final_training=final_training,
        temporal_split_year=2021 if temporal else None,
    )
    groups = dm.group_ids if expanded else dm.ESMEDataset.patient_ids
    partitions = [set(groups[idx]) for idx in dm._partition_indices]
    assert all(
        partitions[i].isdisjoint(partitions[j]) for i in range(4) for j in range(i)
    )
    assert partitions[0] and partitions[1] and partitions[3]
    assert len(set.union(*partitions)) == 30
    if expanded:
        assert (dm.ESMEDataset.mask.sum(1) == 1).all()
    assert len(dm.data_manifest["dataset_sha256"]) == 64


def test_validation_test_values_cannot_fit_scaler_or_grid():
    dynamic, static = frames()
    dm = data_module(dynamic, static)
    training = dm.data_manifest["splits"]["train"]
    changed = dynamic.copy()
    not_training = ~changed.usubjid.isin(training)
    changed.loc[not_training, "X_marker"] = 10000
    changed.loc[not_training, "Y_onset_to_death"] = 10000
    changed_static = static.copy()
    changed_static.loc[~static.usubjid.isin(training), "X_age"] = 10000
    dm2 = data_module(changed, changed_static)
    assert dm.data_manifest["splits"] == dm2.data_manifest["splits"]
    assert dm.data_manifest["scaler"] == dm2.data_manifest["scaler"]
    assert torch.equal(dm.interval_bounds, dm2.interval_bounds)
    train_idx = dm._partition_indices[0]
    assert torch.equal(dm.ESMEDataset.X[train_idx], dm2.ESMEDataset.X[train_idx])
    assert dm.data_manifest["dataset_sha256"] != dm2.data_manifest["dataset_sha256"]


def test_duplicate_line_is_not_an_extra_timestep_and_conflicts_fail():
    dynamic, static = frames()
    duplicate = pd.concat([dynamic, dynamic.iloc[[0]]], ignore_index=True)
    dm = data_module(duplicate, static)
    assert torch.equal(dm.ESMEDataset.X, data_module(dynamic, static).ESMEDataset.X)
    duplicate.loc[len(duplicate) - 1, "X_marker"] = -100
    with pytest.raises(ValueError, match="duplicate"):
        data_module(duplicate, static)


def assignment_model(labels=None, n_lines=2):
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 180) if labels is None else labels
    n = len(labels)
    x, xs = torch.zeros(n, n_lines, 1), torch.zeros(n, 1)
    p, d = torch.zeros(n, n_lines, 3), torch.zeros(n, n_lines, 1)
    fitted = PropensityOverlapModel(n_lines, 0.1, 2).fit(
        x,
        xs,
        p,
        d,
        labels[:, None].expand(-1, n_lines),
        torch.ones(n, n_lines),
        {line: [0, 1] for line in range(n_lines)},
    )
    return fitted, (x, xs, p, d)


def test_support_is_absolute_assignment_probability_not_eligible_conditional():
    model, (x, xs, p, d) = assignment_model()
    assert model.line_results[0].arms.tolist() == [0, 1, 2]
    gate = model.predict_mask(x, xs, p, d, 3)
    assert not gate[:, :, :2].any()  # absolute .05, not conditional .50
    assert gate[:, :, 2].all()
    rec = TreatmentRecommender(
        SimpleNamespace(n_lines=2, n_treatments=3, x_input_dim=1, p_input_dim=3),
        {0: [0, 1], 1: [0, 1]},
        model,
    )
    assert not rec.patient_support_mask(torch.cat((x, p, d), -1), (xs, xs)).any()


def test_propensity_prefix_matches_full_history_and_keeps_rare_classes():
    model, (x, xs, p, d) = assignment_model(torch.tensor([0] * 20 + [1] * 19 + [2]))
    assert 2 in model.line_results[0].arms
    full = model.predict_mask(x, xs, p, d, 3)
    prefix = model.predict_mask(x[:, :1], xs, p[:, :1], d[:, :1], 3)
    assert prefix.shape == (40, 1, 3)
    assert torch.equal(full[:, :1], prefix)


def test_end_to_end_online_recommender_matches_full_history_prefix():
    model = small_model()
    xpd, static, treatments = model_inputs()
    model.recommendable_treatments_per_line = {0: [0, 1, 2], 1: [0, 1, 2]}
    model.recommendation_propensity_model = PropensityOverlapModel(2, 0.1, 2).fit(
        xpd[:, :, :2],
        static[0],
        xpd[:, :, 2:5],
        xpd[:, :, 5:],
        treatments,
        torch.ones(8, 2),
        model.recommendable_treatments_per_line,
    )
    rec = TreatmentRecommender.from_model(model)
    full_rmst, full_mask = rec.arm_rmst(xpd, static, treatments)
    prefix_rmst, prefix_mask = rec.arm_rmst(xpd[:, :1], static, treatments[:, :1])
    assert torch.equal(full_mask[:, :1], prefix_mask)
    assert torch.equal(full_rmst[:, :1], prefix_rmst)


@pytest.mark.parametrize(
    "attribute, values",
    [
        (
            "data_manifest",
            (
                {"protocol_version": 2, "hash": "a"},
                {"protocol_version": 2, "hash": "b"},
            ),
        ),
        ("calibration_status", ("joint_nll_only", "posthoc_reference_bin_fit")),
    ],
)
def test_ensemble_refuses_provenance_and_calibration_mismatch(attribute, values):
    models = [small_model(), small_model()]
    for model, value in zip(models, values):
        setattr(model, attribute, value)
        model.recommendable_treatments_per_line = {0: [0, 1], 1: [0, 1]}
        model.recommendation_propensity_model = assignment_model()[0]
    with pytest.raises(IncompatibleMemberError, match="differ"):
        check_members([TreatmentRecommender.from_model(m) for m in models])


def test_missing_support_fails_closed():
    model = SimpleNamespace(n_lines=2, n_treatments=3)
    rec = TreatmentRecommender(model, {0: [0, 1]}, None)
    assert not rec.patient_support_mask(
        torch.zeros(2, 1, 1), (torch.zeros(2, 1),) * 2
    ).any()
    assert not TreatmentRecommender(model, {}).action_mask().any()
    with pytest.raises(ValueError, match="missing"):
        TreatmentRecommender(model).action_mask()


def test_brier_matches_manual_event_and_survival_ipcw():
    model = small_model()
    train_times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    train_events = torch.tensor([True, False, True, False, True, True])
    times = torch.tensor([1.0, 2.0, 3.0, 5.0])
    events = torch.tensor([True, False, True, False])
    survival = torch.full((4, 7), 0.5)
    survival[:, 0] = 1.0
    _, actual, _ = model.eval_brier_score_ipcw(
        train_events,
        train_times,
        events,
        times,
        survival,
        tmax=3.0,
        device=torch.device("cpu"),
    )
    grid = torch.linspace(0.0, 3.0, model.brier_integration_step)
    weights = get_ipcw(train_events, train_times, new_time=times)
    horizon_weights = get_ipcw(train_events, train_times, new_time=grid)
    expected = torch.stack(
        [
            (
                s.square() * (events & (times <= tau)) * weights
                + (1 - s).square() * (times > tau) * w
            ).mean()
            for tau, w, s in zip(
                grid,
                horizon_weights,
                model.eval_factual_survival(survival, grid, "cpu").T,
            )
        ]
    )
    assert torch.allclose(actual, expected)


def test_member_loader_does_not_invent_eligibility_or_accept_legacy_propensity(
    tmp_path,
):
    model = small_model()
    propensity, _ = assignment_model()
    model.recommendation_propensity_model = propensity
    model.data_manifest = {"protocol_version": 2}
    model.recommendable_treatments_per_line = {}
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
    }
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    assert not load_member(path).recommender.action_mask().any()
    del propensity.assignment_scope
    torch.save(checkpoint, path)
    with pytest.raises(IncompatibleMemberError, match="legacy conditional"):
        load_member(path)


def test_lightning_fit_smoke_uses_separate_validation_and_saves_protocol(tmp_path):
    import lightning as L

    dm = data_module()
    dims = dm.get_data_dimensions()
    template = small_model()
    model = DynaSurvCausalOnline(
        x_input_dim=dims["x_input_dim"],
        x_static_dim=dims["x_static_dim"],
        p_input_dim=dims["p_input_dim"],
        p_static_dim=dims["p_static_dim"],
        n_treatments=dims["p_input_dim"],
        output_length=6,
        interval_bounds=dm.interval_bounds,
        n_lines=2,
        arch=template.arch,
        training=template.training_config,
        evaluation=template.eval_config,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=1,
    )
    trainer.fit(model, datamodule=dm)
    assert "val_loss" in trainer.callback_metrics
    assert model.data_manifest == dm.data_manifest
    assert model.recommendation_propensity_model.assignment_scope == "all_observed"
    path = tmp_path / "smoke.ckpt"
    trainer.save_checkpoint(path)
    restored = load_dynasurv_checkpoint(path)
    assert restored.data_manifest == dm.data_manifest
    assert restored.use_static_features


def truth_fixture():
    return pd.DataFrame(
        dict(
            lineid=[1] * 4,
            arm_idx=[0, 1, 0, 1],
            eta__A=[0.0] * 4,
            eta__B=[0.2] * 4,
            weibull_k=[1.0] * 4,
            weibull_lambda=[5.0] * 4,
            latent_time=[1.0, 3.0, 5.0, 8.0],
            obs_time=[0.1] * 4,
            event=[0] * 4,
        )
    )


def test_semisynthetic_exact_scores_ignore_observed_censoring_and_keep_horizon():
    frame = truth_fixture()
    grid = np.arange(0.0, 4.1, 0.1)
    truth = build_truth(frame, ("A", "B"), grid, [4.0])
    oracle = oracle_prediction(frame, truth.arms, grid)
    wrong = Prediction(np.full_like(oracle.survival, 0.5))
    metrics = factual_metrics({"oracle": oracle, "wrong": wrong}, truth).set_index(
        "predictor"
    )
    factual_p = truth.survival[np.arange(4), truth.factual, -1]
    assert metrics.loc["oracle", "brier"] == pytest.approx(
        np.mean(factual_p * (1 - factual_p))
    )
    assert metrics.loc["wrong", "brier"] == pytest.approx(0.25)
    assert (metrics.tau_used == 4.0).all()  # not silently clipped to .0999
    changed = frame.assign(obs_time=100.0, event=1)
    other = factual_metrics(
        {"oracle": oracle, "wrong": wrong},
        build_truth(changed, truth.arms, grid, [4.0]),
    )
    pd.testing.assert_frame_equal(metrics.reset_index(), other)


def aggregate_module():
    path = Path(__file__).parents[1] / "scripts/semisynthetic/aggregate.py"
    spec = importlib.util.spec_from_file_location("semisynthetic_aggregate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregation_uses_fixed_rule_rejects_legacy_by_default_and_reports_km(tmp_path):
    module = aggregate_module()
    frame = truth_fixture()
    grid = np.arange(0.0, 4.1, 0.1)
    truth = build_truth(frame, ("A", "B"), grid, [4.0])
    oracle = oracle_prediction(frame, truth.arms, grid)
    tables = evaluate(
        {"oracle": oracle, "dynasurv": oracle, "naive_km": oracle},
        truth,
        np.ones((1, 2), bool),
        frame,
    )
    run = tmp_path / "gamma/1.0/rep0/seed_0"
    for name in ("eval_bestCI", "eval_v2_test_val_loss", "eval_v2_test_bestCI"):
        directory = run / name
        directory.mkdir(parents=True)
        for table, data in tables.items():
            data.to_csv(directory / f"{table}.csv", index=False)
        if "v2" in name:
            (directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "evaluation_protocol": 2,
                        "split": "test",
                        "checkpoint_kind": name.removeprefix("eval_v2_test_"),
                        "factual_estimator": "exact_expected_and_uncensored_latent",
                    }
                )
            )
    raw = module.load_raw(tmp_path, kind="val_loss")
    assert set(raw["effect"].eval_kind) == {"val_loss"}
    assert set(raw["effect"].evaluation_protocol) == {2}
    comparison = module.comparator_summary(raw)
    assert set(comparison.predictor) == {"dynasurv", "oracle", "naive_km"}
    assert "mean_pair_pehe_mean" in comparison
    legacy = module.load_raw(tmp_path, kind="bestCI", legacy_exploratory=True)
    assert set(legacy["effect"].evaluation_protocol) == {1}
    with pytest.raises(ValueError, match="No evaluations"):
        module.load_raw(tmp_path, kind="final_epoch")


def test_hpo_refuses_legacy_trials_and_changed_protocol():
    import optuna

    path = Path(__file__).parents[1] / "scripts/hyperopt/run_optuna.py"
    spec = importlib.util.spec_from_file_location("hpo_protocol_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluation = EvalConfig((3.0, 3.0))
    legacy = optuna.create_study()
    legacy.add_trial(optuna.trial.create_trial(value=0.8))
    with pytest.raises(ValueError, match="legacy"):
        module.require_study_protocol(legacy, {"protocol_version": 2}, evaluation)
    fresh = optuna.create_study()
    manifest = {"protocol_version": 2, "dataset_sha256": "a"}
    module.require_study_protocol(fresh, manifest, evaluation)
    fresh.add_trial(optuna.trial.create_trial(value=0.6))
    module.require_study_protocol(fresh, manifest, evaluation)  # valid resume
    with pytest.raises(ValueError, match="different"):
        module.require_study_protocol(
            fresh, manifest | {"dataset_sha256": "b"}, evaluation
        )
