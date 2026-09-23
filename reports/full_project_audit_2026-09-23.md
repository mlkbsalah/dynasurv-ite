# DynaSurv project audit

Audit date: 23 September 2026

Scope: model and its active data/training pipeline; recommendation package; semisynthetic validation; current second manuscript draft.

Source snapshot: Git HEAD `317ec08b7f5dc049d321fbd65278e087c3183b71`, including the existing working-tree edits. No implementation, manuscript, data, or experiment outputs were changed by this audit.

## Overall assessment

**The architecture is suitable as a research model for survival prediction at successive treatment decisions. The current project does not yet establish reliable individualized causal treatment recommendations.** The main obstacles are omitted available covariates, evaluation reuse, incorrect or inappropriate censoring weights, and a gap between the recommendation safeguards described and those actually validated.

| Component | Assessment |
|---|---|
| Survival model | Sensible history-based, discrete-hazard architecture; important input, training, and evaluation defects remain. |
| Recommendation system | Well-structured research policy layer with useful support checks and abstention; its “confidence” is training stability, and several edge cases misstate support or superiority. |
| Semisynthetic validation | Substantial, functioning benchmark with reproducible stored results; presently exploratory because selection uses the evaluation cohort and some factual metrics are biased. |
| Manuscript | Substantial working draft that compiles cleanly; scientific corrections and replacement experiments are needed before submission. |

The most urgent work is to repair the train/validation/test design and survival metrics, restore valid static predictors, and rerun the comparisons. Editing the manuscript around the existing numbers alone would leave the main evidential problems unresolved.

Severity terminology: **P1** changes the interpretation of primary results or defeats a stated system requirement; **P2** is a material limitation or conditional defect; **P3** is maintainability/reporting work. “Verified” means supported by inspected code, saved artifacts, or a numerical reproduction. Methodological limitations are distinguished from arithmetic bugs.

## 1. Model suitability and data pipeline

### What the model gets right

The encoder conditions on observed covariates, prior treatment embeddings, and elapsed time; each treatment has its own output coordinates for discrete hazards. This is a reasonable way to share information across related treatment decisions without imposing proportional hazards. The main loss masks nonexistent patient-lines, and predictions produce monotone survival curves with an explicit survival probability of one at time zero.

Crucially, the current treatment is embedded for the **next** step, rather than used to form the current decision representation. I verified that changing the current treatment index leaves that line’s entire vector of treatment predictions unchanged, while changing the subsequent representation. This supports the intended comparison of alternative current treatments given the same prior history. See [encoder:249](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/embedding_C_LSTM_ITE.py:249) and [encoder:258](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/embedding_C_LSTM_ITE.py:258).

The code also excludes the previously identified `X_onset_to_progression` leak by default. The active real-data V2 parquet already lacks that column. This is a useful protection, but it does not certify the timing of the other features.

The appropriate target is **survival from the current line start under a current treatment choice, with subsequent care following the observed care system**, conditional on reaching that line. This is neither a simulator of whole treatment sequences nor proof that repeatedly following its recommendations optimizes lifetime survival.

### Current data actually entering the model

I reconstructed the configured 2018+ HR+/HER2− cohort from the local parquet files.

| Stage or quantity | Verified value |
|---|---:|
| Eligible before static-data join | 6,256 patients; 13,257 records |
| After static-data inner join | 6,191 patients; 13,117 records |
| Training / temporal holdout patients | 4,214 / 1,977 |
| Training records by tensor line | 4,214 / 2,561 / 1,731 / 1,157 |
| Holdout records by tensor line | 1,977 / 841 / 438 / 198 |
| Dynamic inputs | 72: 71 existing X columns plus calendar time |
| Static patient / static treatment-history inputs | 16 / 17 |
| Treatment categories / survival intervals | 11 / 100 |
| Survival grid | 0–72.509 months; approximately 0.725 months per interval |
| Missing values in constructed dynamic/static tensors | 0 |
| Duplicate patient-line records | 1 identical duplicate |

Absence of missing values in these imputed tensors does not establish valid imputation or measurement timing.

### M1 — P1, verified: all static predictors are ignored

[Model:221](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:221) receives both static tensors, then initializes hidden, cell, and treatment states to zero. The commented projections are never applied, and the static tensors have no other predictive use.

Available but ignored columns include selection age, primary-diagnosis age, menopause, BMI, family history, and primary-treatment history. Perturbing **every static value** by large random amounts changed model logits by exactly **0.0**.

This directly weakens prognostic adjustment and potentially removes recorded confounders. The external propensity model sees some static information, but a support filter does not insert those omitted variables into the outcome regression. Whether every candidate static measurement was available at the relevant decision time still requires a provenance check.

**Improve:** incorporate verified pretreatment static variables into the encoder or output representation, with explicit feature lists and a static-feature ablation. Correct the manuscript’s claim that age is unavailable: the current problem is primarily that the model does not use available age variables.

### M2 — P1, verified: there is no untouched test cohort in the main workflow

[Datamodule:834](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/datamodule_cv.py:834) assigns the temporal holdout to both `val_dataset` and `test_dataset`. [Training:42](/Users/malek/TheLAB/DynaSurv/scripts/TrainDynasurvCausal.py:42) always selects this mode. Epoch selection, early stopping, final testing, and recommendation summaries therefore reuse the same patients. Hyperparameter optimization also explicitly optimizes on this temporal holdout: [Optuna:124](/Users/malek/TheLAB/DynaSurv/scripts/hyperopt/run_optuna.py:124).

There is a separate defect in the real-data CV path: [datamodule:818](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/datamodule_cv.py:818) constructs folds over the **entire dataset**, not the training pool after reserving the holdout. A numerical check found that all 1,977 nominal holdout patients reappear somewhere in the training, validation, or early-stopping partitions. The early-stop split also lacks an explicit `random_state`. The semisynthetic subclass fixes this CV-pool error, but its production training path still inherits validation/test reuse.

**Improve:** reserve a patient-disjoint final evaluation set; perform hyperparameter, epoch, calibration, and recommendation-threshold selection entirely within development data. In temporal experiments, define the calendar roles of all partitions explicitly. Existing temporal results are development-holdout results, not independent test estimates.

### M3 — P1, verified: the main IPCW Brier score omits event-time weights

[Model:979](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:979) computes weights at evaluation times and passes only `weight_new_time` to TorchSurv. It omits `weight`, the weights at each subject’s observed event time. In the installed TorchSurv 0.1.6, `_update_brier_score_weight()` then sets **both weight arrays to ones**, overriding even the supplied evaluation-time weights. The resulting score is not IPCW-adjusted.

Both the observed-event and surviving-control error terms consequently lack their required censoring adjustment. A controlled example returned Brier scores **[0.25, 0.25]** through the current calling convention, versus **[0.30, 0.375]** when both required weights were supplied. The same omission occurs in [Cox:245](/Users/malek/TheLAB/DynaSurv/scripts/Cox_baseline.py:245), [DeepSurv:402](/Users/malek/TheLAB/DynaSurv/scripts/Deepsurv_baseline.py:402), and [RSF:243](/Users/malek/TheLAB/DynaSurv/scripts/RSF_baseline.py:243).

I checked the installed TorchSurv implementation as well as its [official Brier-score documentation](https://opensource.nibr.com/torchsurv/_autosummary/torchsurv.metrics.brier_score.html). This is a concrete implementation defect, separate from the temporal censoring-distribution problem discussed in section 3.

**Improve:** supply weights at both observed and evaluation times, test against a trusted numerical reference, and regenerate IBS tables and any checkpoints selected using `bestIBS`. A bug shared by all baselines does not guarantee their ranking remains unchanged.

### M4 — P2, verified: preprocessing is fitted before splitting and is not a self-contained checkpoint artifact

[Datamodule:659](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/datamodule_cv.py:659) fits scaling using the full merged cohort, including the temporal holdout. Continuous-column detection, treatment vocabulary, and the maximum-outcome-derived grid also precede splitting.

For calendar time, the stored mean was **35.126 months**, whereas the training-only mean was **29.214**. This is real test-information use, although the resulting performance impact has not been quantified. More practically, rebuilding the datamodule on changed data changes preprocessing even when loading unchanged weights.

Checkpoint support state is persisted, which is good, but feature names/order, fitted preprocessing, treatment vocabulary, and split membership are not fully bound to that checkpoint; see [model:1164](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:1164).

**Improve:** split first; fit transformations, vocabulary, and time grid on training data; serialize the complete input schema and preprocessing with the model. Audit how the upstream imputed parquet was constructed, including whether imputation used future observations or evaluation subjects.

### M5 — P2, verified: an exact duplicate becomes a fictitious second treatment line

The real-data join/sort at [datamodule:360](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/datamodule_cv.py:360) does not check uniqueness. Sequence construction uses row position, not an explicit validated line map. One retained patient has the sequence **[1, 1]**, consisting of two identical rows. The second copy is treated as tensor line 2 and enters its loss/support counts.

The semisynthetic loader explicitly removes this duplicate at [drivers:51](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/drivers.py:51). This explains its 13,116 records versus the real model’s 13,117.

The immediate numerical impact is small, but the integrity failure is real.

**Improve:** enforce one row per patient/line and contiguous line numbering before tensorization; remove only proven exact duplicates and reject conflicting duplicates. Report the 65 patients excluded by the static-data join rather than presenting the pre-join count as the final model cohort.

### M6 — P2, verified behavior: within-interval censoring is rounded upward

[Time transformation:79](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/utils.py:79) maps both events and censorings to the containing interval. [Loss:28](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/metrics/survival_loss.py:28) includes survival through that interval for censored observations.

For a patient censored at month 5 on bounds [0, 10, 20], the loss charges survival through the first 10-month interval. With hazard 0.5 it contributes 0.693, despite no complete interval being observed. This grants information beyond the censoring time. It is a discretization approximation whose impact depends on grid width; it should not be silently treated as exact continuous-time censoring likelihood.

**Improve:** define and validate an event/censoring discretization convention, for example right-round events and left-round censorings, or use a likelihood accounting for partial exposure. The [pycox reference implementation](https://raw.githubusercontent.com/havakv/pycox/master/pycox/preprocessing/discretization.py) explicitly distinguishes the two. Verify boundary cases and quantify the effect at the current 0.725-month grid.

### M7 — P2, verified structure: padded rows affect BatchNorm during training

[MLP:30](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/mlp.py:30) uses BatchNorm. [Model:192](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:192) processes every patient at every padded step; masking occurs only in the losses.

Consequently, patients without a later line still participate in that line’s feature/output normalization. This is especially material when only 1,157 of 4,214 training patients have a fourth line. The same BatchNorm modules also share running statistics across lines with different case mix and padding fractions.

**Improve:** normalize independently of batch padding, such as with an appropriately validated LayerNorm design, or process only active rows with masks. Add a test that changing padding cannot change valid-row predictions or normalization statistics. Assess this alongside the observed line-specific calibration differences.

### M8 — P2: the causal regularization and training objective need honest characterization

All three balancing coefficients in [best_config.json](/Users/malek/TheLAB/DynaSurv/configs/best_config.json) are zero. The gradient-reversal implementation is correctly signed, but there is currently no evidence that active balancing produces the reported results. This does **not** mean outcome regression cannot estimate causal effects under adequate adjustment; it means gains cannot be attributed to balancing.

Simply increasing the coefficients is insufficient. Under the actual unshuffled training batches and a minimum of 16 observations per treatment group, eligible IPM comparisons were:

| Line | Usable comparisons / candidate batch comparisons |
|---|---:|
| 1 | 54 / 198 |
| 2 | 31 / 330 |
| 3 | 1 / 99 |
| 4 | 0 / 33 |

See [IPM:63](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/metrics/ipm.py:63), [training loader:864](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/data/datamodule_cv.py:864). A nonzero coefficient alone would still provide essentially no balancing at the last two lines under these batches.

The survival objective also takes per-line means and then weights them inversely by their batch counts: [model:714](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:714). In the inspected pass, mean line weights were approximately **11.4%, 18.8%, 27.8%, 42.0%**. This deliberately favors sparse late lines and creates a batch-dependent objective. It is not the ordinary pooled likelihood or equal-line average.

**Improve:** state the intended target weighting, compare it with fixed/equal-line weighting, and tune regularization only with usable group sizes and appropriate batches. Compare a smaller model and current-line-only baseline to establish the value of recurrence and depth.

Lower-priority engineering issues include unreachable duplicated loss code after [model:732](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/model/dynasurv_causal_online.py:732), an unused encoder output MLP, and multiple stale model/training variants. The extra “layers” reuse recurrent gate weights and one state rather than implementing a conventional stacked LSTM; describe the architecture precisely. The old evaluator also has untested public paths, including accessing plotting axes when `plot=False` at [evaluator:165](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/evaluation/evaluator.py:165).

### Suitability decision

Retain this architecture as a research candidate. Before accepting it for these data, require valid static-feature inclusion, a documented pretreatment feature-timing audit, corrected metrics and independent evaluation, and comparisons demonstrating benefit over simpler outcome models. The observational data and learned representation alone cannot establish absence of hidden confounding; balancing and propensity thresholds do not supply that guarantee.

## 2. Recommendation-system audit

The core sequence is coherent: predict each arm’s survival/RMST; apply line-level and patient-level eligibility; combine seed-specific predictions; compare paired RMST differences; emit a decision or unresolved set. Useful safeguards already include finite-value checks, explicit all-masked handling, paired differences, observed-line output masks, and checkpoint dimension/grid checks.

### R1 — P1, verified: the propensity gate estimates a conditional treatment probability

[Propensity fitting:99](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/evaluation/propensity_overlap.py:99) discards patients whose factual treatment is outside the eligible-arm set before fitting logistic regression. It estimates:

\[
P(A=a\mid H,\ A\in\mathcal A_{\mathrm{eligible}})
\]

rather than \(P(A=a\mid H)\). It therefore cannot detect a patient with very low probability of receiving **any** eligible treatment.

With two to four eligible arms and a 0.01 floor, at least one arm necessarily passes each member’s filter because these conditional probabilities sum to one. Intersecting different member masks could still remove all arms, but the current M40 masks are identical, with zero recorded disagreement. Thus its absence of `NO_SUPPORT` outputs is not evidence of universal absolute overlap. Out-of-fold fitting is useful but does not repair the target probability or establish calibration.

**Improve:** fit assignment across all observed classes and apply eligibility afterward, or model eligible-set membership separately and recover absolute probabilities. Validate calibration, overlap, and effective sample size in relevant patient subgroups.

### R2 — P1 for interpretation, verified: a sole eligible arm is automatically “confident”

[Ensemble:455](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/ensemble.py:455) gives the only available arm every vote. With no rival its candidate set is a singleton, so [ensemble:478](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/ensemble.py:478) labels it `CONFIDENT`, even if the arm’s own RMST varies greatly across members.

This affects interpretation of the actual latest 40-member bestCALIB output:

| Line | Patient-lines | Reported confident | Confident with just one eligible arm | Confident with a comparative choice |
|---|---:|---:|---:|---:|
| 1 | 1,977 | 1,409 (71.27%) | 6 | 1,403 |
| 2 | 841 | 371 (44.11%) | 3 | 368 |
| 3 | 438 | 5 (1.14%) | 5 | **0** |
| 4 | 198 | 2 (1.01%) | 2 | **0** |

Source: [latest recommendation parquet](/Users/malek/TheLAB/DynaSurv/reports/recommendations/HR+HER2-_4lines/20260921_175204_bestCALIB_M40/recommendations.parquet).

**Improve:** add `ONLY_SUPPORTED_OPTION` as a distinct outcome and reserve comparative-superiority language for decisions with at least two eligible candidates. The current later-line result is that the model does not confidently separate the available chemotherapy alternatives.

### R3 — P2, methodological: seed agreement is not calibrated uncertainty

[Ensemble:444](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/ensemble.py:444) calculates standard deviation across repeated fits to the same patients; `p_best` is a fraction of argmax votes. The “LCB” is mean minus a configured multiple of this standard deviation, with no demonstrated coverage probability.

These quantities measure training variability. They omit sampling uncertainty and can agree under shared confounding, feature omission, or model misspecification. Likewise, failure to separate two arms by the margin does not establish statistical equivalence.

**Improve:** call these seed-spread/stability measures and the output an unresolved candidate set. Evaluate error/regret versus decision coverage on fresh simulations. Add patient-level resampling or external cohorts where useful, while keeping hidden-confounding sensitivity separate. The manuscript’s current caveat about shared bias is worth preserving.

### R4 — P1 for the online API, verified: shortened histories crash the propensity stage

The recommender accepts an observed number of lines, but [propensity prediction:160](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/evaluation/propensity_overlap.py:160) loops through all fitted line models. A one-line input using an actual current bestCALIB checkpoint fails with:

```
ValueError: X has 100 features, but LogisticRegression is expecting 172 features as input.
```

The full padded holdout workflow works; a natural online prefix call does not.

**Improve:** evaluate only available lines, size the mask to the provided prefix, and test prefix-versus-padded inference equivalence for every decision point.

### R5 — P2, verified: missing support can silently enable unrestricted advice

[Action mask:77](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/recommender.py:77) treats both `None` and an empty dictionary as “every treatment eligible.” Missing propensity objects or missing fitted lines also leave global eligibility in force. Propensity fitting silently skips some unsupported fitting conditions.

These behaviors support legacy checkpoints, but unknown support and affirmative support are different states.

**Improve:** fail closed for production recommendation; distinguish empty eligible sets, absent metadata, failed propensity fitting, and deliberately unrestricted research mode. Include reason codes in outputs.

### R6 — P2, verified edge case: exact ties become confident at zero margin

With every member predicting [5, 5] and `margin_months=0`, deterministic argmax sends all votes to arm 0. The strict comparison `diff_lcb < margin` at [ensemble:471](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/ensemble.py:471) excludes its equally good rival. Arm 0 is reported confident.

The existing test currently encodes this behavior. The shipped one-month margin avoids this exact case.

**Improve:** preserve tied alternatives and define how ties contribute to votes.

### R7 — P2: member compatibility and preprocessing provenance are incomplete

Current checks compare dimensions, architecture, grids and numeric support sets, but cannot detect different same-width feature orderings, treatment meanings, imputation/scaling, or original split membership. [Member assembly:185](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/pipeline.py:185) rebuilds current data settings; member loading replaces historical eligibility with that current setting. Non-temporal inference also uses a fixed split seed of zero.

No mixed-seed identity error was found in the current M40 artifact: it contains 40 distinct seed IDs. This is a protection gap for future or changed runs.

**Improve:** persist and compare feature names/order, treatment vocabulary, preprocessing, cohort/split hashes, source/config snapshots, and unique checkpoint identities. A Git HEAD and config pathname alone do not capture a dirty worktree or later config edits.

### R8 — P1 for manuscript accuracy: bestCALIB is not post-hoc recalibration

The training and recommendation scripts do not call `HazardCalibrator`. [Checkpoint discovery:78](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/recommendation/ensemble.py:78) explicitly excludes recalibrated filenames. Current recommendations use raw bestCALIB checkpoints: selecting the epoch with the smallest marginal calibration gap is different from performing the post-hoc calibration described in the paper.

Furthermore, the rank-preservation claim in [calibrator:67](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/evaluation/hazard_calibration.py:67) is false for general interval hazards. Example:

- Hazards A = [0.01, 0.75], B = [0.5, 0.5].
- Final survival initially: A = 0.2475, B = 0.25.
- Divide all logits by temperature 2: A = 0.332598, B = 0.25.

A monotone interval-wise transformation reversed the survival ordering. C-index invariance and fixed ordered-bin membership therefore cannot be assumed.

**Improve:** distinguish joint training, epoch selection, and post-hoc calibration. If recalibration is adopted, fit it within development data and re-evaluate discrimination, treatment rankings, and final decisions. Marginal factual calibration does not establish all-arm counterfactual calibration.

### R9 — P2: horizon robustness includes extrapolation and margin effects

The horizon sweep keeps eligibility fixed, which is appropriate for isolating horizon sensitivity. However, the current sweep extends to 36 months when maximum observed holdout time is only 31.97 months at line 2, 28.98 at line 3, and **19.94 at line 4**. Its “known outcome” coverage includes deaths, so nonzero coverage beyond the last observation does not establish an identifiable surviving tail.

The fixed one-month margin also explains much of the changing confident share: line 1 rises from roughly 0.3% at 3 months to 71.27% at 24 months and 82.8% at 36, while leaders agree above 99%.

**Improve:** label unsupported horizons as extrapolation, report per-arm risk-set/censoring support, and distinguish rank stability from the changing ease of clearing an absolute benefit margin. These plots are sensitivity analyses, not validation of every tested horizon.

Finally, checkpoint choice matters: using the same 40 seeds with `val_loss` gives confident rates of **80.98%, 51.37%, 1.14%, 1.01%**, versus the bestCALIB rates above. The checked-in default still selects `val_loss`. Pin the intended selection rule before a final experiment.

## 3. Semisynthetic validation

### What is sound

The benchmark is real and operational, rather than a placeholder. All **120 evaluation directories** were present: 30 runs × four checkpoint kinds, each containing curve, effect, policy and factual tables. The published aggregate numbers reproduce from these outputs.

Useful design choices include strict driver definitions, separate random streams, common random draws across sweeps, analytical Weibull survival/RMST, endpoint-only masks for expanded prefixes, original-patient grouping in the custom split methods, and training-only arm-support counts. The oracle has zero curve error/regret, with only approximately \(1.9\times10^{-5}\) months of numerical integration residual in effect scores.

### S1 — P1, verified: selection reuses the reported evaluation data twice

The semisynthetic training command calls the same `final_training=True` path described in M2. Its grouped CV correction is not used by that path. Then [aggregation:151](/Users/malek/TheLAB/DynaSurv/scripts/semisynthetic/aggregate.py:151) chooses the checkpoint kind with the lowest **oracle regret on the same evaluation runs** and reports sweeps for that choice.

The tables are numerically reproducible, but the selected performance is exploratory and potentially optimistic.

**Improve:** use original-patient-grouped training/validation/test partitions, select the checkpoint rule on development replicates, and assess that fixed rule on new data/replicates. Preserve the existing comparisons as hypothesis-generating results.

### S2 — P1, numerically verified: training-cohort censoring weights bias factual scores

Administrative censoring is the database cutoff minus actual line-start date: [censoring:41](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/censoring.py:41). Recent holdout patients therefore have much shorter follow-up than older training patients. Nevertheless [evaluation:258](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/evaluate.py:258) supplies the older training cohort’s marginal censoring distribution to the IPCW estimator.

For the available reference truth dataset (`data/semisynthetic/gamma/1.0/rep0`, holdout defined by first entry in 2021 or later), even exact oracle predictions yield:

| Line / horizon | Training administrative follow-up reaches horizon | Holdout reaches horizon | Exact expected oracle Brier | Current factual evaluator Brier |
|---|---:|---:|---:|---:|
| 1 / 24 months | 100.0% | 67.0% | 0.1633 | 0.1358 |
| 2 / 18 months | 91.4% | 46.0% | 0.1923 | 0.1382 |
| 3 / 12 months | 92.5% | 53.0% | 0.1925 | 0.1349 |
| 4 / 12 months | 88.6% | 39.4% | 0.1949 | 0.1057 |

The corresponding uncensored sampled-outcome Brier scores were 0.1584, 0.1939, 0.1944 and 0.1866. The discrepancy is substantial, not merely a theoretical concern.

The assertion at [censoring:7](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/censoring.py:7) that censoring is unconditionally independent of treatment/outcome is unsupported: administrative follow-up depends on calendar time, which affects assignment and is associated with cohort characteristics. Marginal censoring weighting requires assumptions stronger than simply generating independent dropout. See the [official scikit-survival assumptions](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html).

**Improve:** make exact expected scores and uncensored latent-outcome scores primary simulation checks. Separately develop evaluation-population-appropriate censoring adjustment with explicit support restrictions. Simply replacing training KM with test KM does not automatically repair covariate-dependent censoring.

This issue affects factual IPCW scores. It does **not** invalidate directly computed oracle RMST/PEHE/regret. The semisynthetic Brier API itself handles event-time weights correctly; its error is distinct from M3.

### S3 — P1 for conclusions: ranking benefit and effect estimation give different answers

The current summary omits the strongest readily available caution: at the selected bestCI checkpoint, DynaSurv’s mean pair PEHE is **worse than naive per-arm Kaplan–Meier in 9 of 10 cells**.

| Cell | DynaSurv PEHE, months | Naive KM PEHE, months |
|---|---:|---:|
| Reference observed confounding = 1 | 1.761 | 1.704 |
| Hidden-confounding strength = 1 | 2.699 | 2.171 |
| Heterogeneity = 0 | 2.082 | 1.192 |
| Heterogeneity = 2 | 2.644 | 2.773 |

The final row is the only winning cell on this metric. Evidence is retained in [effect_raw.csv](/Users/malek/TheLAB/DynaSurv/reports/semisynthetic_sweep/effect_raw.csv); [aggregation:90](/Users/malek/TheLAB/DynaSurv/scripts/semisynthetic/aggregate.py:90) summarizes DynaSurv effect metrics without the corresponding KM comparison.

Policy ranking is more promising. Reference regret is **0.3356 months** for DynaSurv versus **0.5156** for KM, **0.5135** for the oracle-selected best constant arm, and **0.8007** for random choice.

**Improve:** report both results. These outputs suggest some treatment-ranking value in this DGP, not superior treatment-effect estimation. Add fitted Weibull/Cox, DeepSurv and simple history-aware baselines, plus static/history/balancing ablations. The current benchmark evaluates only DynaSurv, oracle and KM: [evaluation script:79](/Users/malek/TheLAB/DynaSurv/scripts/semisynthetic/evaluate.py:79).

### S4 — P2: the full-information oracle is not the observable-history target

[Oracle:41](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/predictors.py:41) conditions on each patient’s hidden \(u\). The hidden outcome coefficient remains 0.4 even when hidden assignment strength is zero: [DGP config:120](/Users/malek/TheLAB/DynaSurv/configs/semisynthetic/dgp.toml:120). An observed-history model can recover an average over \(u\mid H\), not that inaccessible individual value.

A 40-point Gaussian-quadrature check over the known distribution of \(u\) conditional on liver status found that even the ideal observable-history oracle has RMST RMSE **1.213 months** against the hidden-\(u\) oracle in the reference holdout; pair PEHE floors are approximately **0.141–0.344 months**. These checks use all four DGP arms before support filtering. Best-arm agreement remains 100% because the hidden variable adds a common log-hazard term across arms.

**Improve:** retain the privileged oracle, add an observable-history oracle, and separate irreducible information loss from model estimation error and confounding bias. Zero hidden assignment coupling does not imply zero error against hidden-conditioned curves.

### S5 — P2: the assignment sweep does not hold the outcome mechanism fixed

[Outcome fitting:115](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/outcome.py:115) refits Weibull shape/scale to each newly assigned factual mixture. Dropout rates are then recalibrated. Thus changing assignment also changes the outcome surface and censoring.

For replicate 0, changing observed-confounding strength from 0 to 1.5 changes line-1 Weibull scale from **60.407 to 56.636 months**, and line-4 scale from **19.551 to 17.818**. Line-4 eligible sets also vary between two and three arms, changing the oracle action set and the effect pairs entering summaries.

**Improve:** freeze reference outcome/censoring parameters for a clean assignment-strength experiment, and report a fixed-common-support comparison. Mixture-recalibrated scenarios can be a separate realism analysis, but cannot isolate assignment strength alone.

### S6 — P2: the benchmark does not validate the full recommendation policy

The construction freezes real histories and generates only the endpoint decision/outcome. It does not simulate treatment effects on future covariates, entry into later lines, or subsequent care. Its shared-shape Weibull treatment effects also cannot generate crossing arm survival curves or horizon-dependent optimal arms.

Prediction calls the single-model `TreatmentRecommender.arm_survival()`; [policy evaluation:207](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/evaluate.py:207) then takes its own RMST argmax. It does not evaluate the ensemble’s paired uncertainty, vote threshold, minimum meaningful benefit, or full abstention rule. Every saved policy row has zero abstention.

The semisynthetic inputs additionally copy age and menopause into dynamic columns, explicitly compensating for the model’s ignored statics: [drivers:25](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/drivers.py:25). This is sensible for the simulated design, but means its observable feature set is richer than the active real-data encoder’s.

**Improve:** label the existing exercise single-decision curve/effect/ranking validation. Add actual ensemble-policy evaluations reporting recommendation coverage, regret among recommended patients, harmful-choice rate, singleton-support cases and abstentions. Use more varied outcome mechanisms, including nonproportional hazards; add coherent longitudinal simulations only if making claims about feedback or repeated policy use.

### S7 — P2/P3: uncertainty, provenance and edge cases remain incomplete

Generation and training seeds are coupled by replicate, so their contributions to variability cannot be separated; only three replicates also provide weak evidence for a monotone bias trend. All 30 saved generator manifests have `git_hash: null`; the current aggregation script is untracked. Locally, only reference-cell replicate 0 retained full truth/model-facing parquets; other cells retain manifests and evaluation outputs, limiting independent reconstruction without regeneration.

Additional verified maintenance issues:

- Aggregation matches seed directories but omits the training seed from its tags, which would conflate additional seeds within a replicate.
- “Pooled PEHE” is an average of already-rooted pair errors. The reference value is 1.761 under that definition versus approximately 1.954 for the corresponding pooled root mean square. Define and label the intended statistic.
- Zero eligible arms can produce `-inf` oracle values and invalid random-policy averages: [policy evaluation:189](/Users/malek/TheLAB/DynaSurv/src/CausalSurv/semisynthetic/evaluate.py:189).
- There are no committed semisynthetic tests in the current four test modules.
- Statements that hidden confounding affects bias “not PEHE” are incorrect: PEHE includes squared bias and variance.

**Improve:** preserve source/config/data/checkpoint hashes, add focused generator/split/oracle/support tests, and use separate repeated data and training seeds with paired uncertainty summaries.

## 4. Manuscript: current state and improvements

I reviewed the complete current [second_draft.tex](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex) and included [semisynthetic appendix](/Users/malek/TheLAB/DynaSurv/latex/appendix_semisynthetic.tex). Compilation in a temporary directory succeeded: **23 pages**, no missing figures or unresolved references reported, and no overfull-box errors. The PDF skill informed a visual spot-check of seven representative pages. Layout was generally clean; architecture and calibration labels need enlargement, and the empty title block leaves conspicuous blank space.

Its strongest content is the explicit single-line decision target, the cohort analyses motivating restriction/support, the discussion of seed uncertainty, and the detailed benchmark appendix. It is not submission-ready because the central empirical claims and methods description are not yet aligned.

### D1 — P1: repair the causal factorization and DAG

At [draft:84](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:84), the joint model factors progression and overall survival separately and treats line-specific OS values as separate terminal nodes. For a patient reaching the next line, however, these are rebased views of the same death:

\[
O_k=(\text{start}_{k+1}-\text{start}_k)+O_{k+1}.
\]

The current graph also lacks the future-treatment-to-death paths needed by the total-effect interpretation at [draft:178](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:178).

**Improve:** either use one terminal death process with line-entry/at-risk and state/treatment processes, or remove the unnecessary joint factorization and show a decision-specific DAG. The single-decision exchangeability argument can remain; the displayed joint model cannot justify it as currently written. Include censoring support through the chosen horizon, not just conditional independence.

### D2 — P1: replace outdated or mismatched evidence

The factual baseline table is explicitly from an earlier full-cohort random split: [draft:579](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:579). Current-cohort matched baselines, validated post-calibration plots, and convincing uncertainty estimates are absent. Some named comparator methods do not appear in the table.

The recommendation table accurately reproduces the current M40 artifact. Preserve its provenance, but correct the interpretation of later-line “confidence” and label selection-cohort results appropriately.

The semisynthetic tables are based on real saved experiments, but need the KM effect-error comparison, selection caveats, censoring correction, and observable-oracle distinction described above.

**Improve:** regenerate one consistent set of tables from the corrected pipeline. Every table/figure should identify cohort, split, feature version, checkpoint rule, calibration status, horizons, support rules, and uncertainty unit.

### D3 — P1/P2: reconcile text with the implemented system

Specific corrections:

| Current claim or ambiguity | Required revision |
|---|---|
| Age unavailable: [draft:212](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:212), [draft:668](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:668) | Age exists in static data but is ignored by the encoder; discuss feature timing and inclusion. |
| Final cohort 6,256 / 13,257: [draft:305](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:305) | Give pre-join, post-join and deduplicated counts separately. |
| Post-hoc calibrated curves: [draft:425](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:425) | Describe the actual raw bestCALIB pipeline or implement and evaluate the stated calibration procedure. |
| Calibration preserves C-index | Remove the guarantee; interval-wise monotonicity does not preserve aggregate survival order. |
| Equivalent treatments: [draft:508](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:508) | Use unresolved/not clearly separated candidates. |
| Treatment-prediction AUC supports ignorability: [draft:670](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:670) | Predictable treatment assignment does not establish that all confounders are measured. |
| Chance calendar AUC establishes absence of availability/confounding issues | Treat it as a limited diagnostic, not proof of positivity or stable availability. |
| Observed confounding only increases variance: [draft:641](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:641) | Qualify by correct specification, overlap and finite-sample estimation; weak overlap and regularization can yield bias. |
| Q-learning needs an explicit transition simulator: [draft:183](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:183) | Distinguish model-free regime learning from model-based simulation. |
| Regulatory rollout automatically supplies an unconfounded external check: [draft:659](/Users/malek/TheLAB/DynaSurv/latex/second_draft.tex:659) | Specify and defend the proposed quasi-experimental design and its assumptions. |
| “Buffer time” \(d_k\) equals the recurrence input | Reconcile with the implemented onset-to-onset elapsed time; the data also have a distinct buffer-time feature. |

Remaining broad treatment classes also contain multiple regimen versions. Excluding OTHER/ET+TT does not by itself prove consistency for chemotherapy or endocrine categories. Define the intervention as an explicit regimen or a specified within-category treatment policy.

The appendix should also correct its unconditional-censoring claim, the “bias, not PEHE” statement, and attribution of zero-heterogeneity random-policy regret to administrative censoring; that regret is computed from true RMST and reflects arm main effects.

### D4 — P2/P3: complete the manuscript around a narrower, supportable contribution

The current file contains **36 TODO calls**, empty title/author fields, no abstract, no real bibliography/citation system, an unfinished metrics appendix, and stale architecture/calibration figures. Several placeholders contain substantive unfinished analyses, not merely editorial reminders.

A productive revision sequence is:

1. State the target population and single-decision estimand precisely, with treatment versions, available history, natural downstream care, and treatment/censoring support.
2. Present a patient-flow and split diagram that matches the data actually modeled.
3. Describe the actual tied-gate recurrent architecture, feature handling, loss weighting, and selection/calibration protocol.
4. Report matched factual baselines and ablations, then simulation effect/policy performance, then the real-data recommendation behavior.
5. Show both favorable and unfavorable comparisons. Establish added personalization against a fixed-arm policy instead of relying on recommendation frequency.
6. Move lengthy exploratory cohort details to the appendix where needed, complete citations and the abstract, and enlarge small figure labels.

A defensible present framing is an **exploratory study of treatment ranking under support restrictions using a recurrent survival model**. Claims of validated individualized causal benefit should wait for corrected independent evaluation.

## 5. Recommended work order and completion criteria

| Order | Work | Evidence needed before moving on |
|---|---|---|
| 1 | Correct patient/line integrity, static features and pretreatment feature timing | Explicit cohort/feature manifest; uniqueness and timing checks; no ignored required inputs |
| 2 | Separate training, selection/calibration and final evaluation; fit preprocessing only on training | Patient-disjoint split assertions and frozen schema/scaler/grid/vocabulary |
| 3 | Fix main Brier weighting, censoring evaluation assumptions, and discretization convention | Numerical agreement with reference calculations, including simulations with known truth |
| 4 | Repair support probabilities, online prefix inference, singleton status and tie handling | Focused integration tests using real checkpoint/support objects |
| 5 | Run current-cohort baselines and model ablations | Independent factual discrimination/calibration/RMST results with patient-level uncertainty |
| 6 | Validate the complete recommendation policy in simulation | Coverage–regret curves, comparator effect errors, observable oracle, fresh selection-free replicates |
| 7 | Freeze experiment provenance and revise the manuscript | All numbers generated from identified runs; corrected methods/claims; completed figures and citations |

Restore the correctness of the measurement process before large architecture searches or large balancing sweeps. Otherwise additional runs will optimize or report the same flawed criteria.

## 6. Verification performed and audit limits

Performed:

- Read the requested model, its encoder/loss/data/config/training dependencies, all recommendation modules and relevant CLIs, the semisynthetic generator/evaluator/aggregation chain, the current manuscript and appendix.
- Reconstructed the active cohort and feature dimensions; checked missing tensor values, duplicates, join attrition, split intersections, and training-batch support.
- Numerically demonstrated ignored static features, current-treatment invariance, the Brier weighting omission, censoring-bin behavior, and hazard-temperature ranking reversal.
- Ran the existing test suite: **33 passed**. These tests mainly cover RMST, recommendation tensor rules, horizon sweeps and run discovery; they do not cover the major split/metric/static-input/semisynthetic issues above.
- Reproduced online-prefix failure using an actual checkpoint and checked singleton/tie/support edge cases.
- Recomputed recommendation counts from current M40 artifacts and cross-checked the checkpoint/horizon sensitivity outputs.
- Checked all saved semisynthetic evaluation tables and numerically evaluated the available reference truth dataset.
- Compiled the draft in a temporary location and visually reviewed representative pages.

Limits:

- No model was retrained, no hyperparameter search was launched, and no existing results were regenerated or overwritten.
- This audit does not establish the original clinical measurement time or imputation provenance of every X/static feature. That requires inspecting the upstream extraction process and dated source records.
- Detailed truth-based numerical checks outside the reference semisynthetic cell require regeneration or recovery of those cells’ parquet files.
- The report identifies implementation defects and limits of the evidence; it does not infer clinical efficacy from recommendation counts.

The only repository artifact added by the audit is this report. Existing manuscript/report edits and the untracked aggregation script were preserved.
