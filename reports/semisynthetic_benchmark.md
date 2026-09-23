# Semi-synthetic counterfactual benchmark for DynaSurv

> Audit correction, 2026-09-23: the numerical results below are **historical/exploratory**, not corrected independent-test validation. Their holdout was reused for early stopping and checkpoint-kind selection; marginal training-cohort IPCW scores are also affected by temporal censoring shift. Protocol v2 separates training/validation/test by original patient, uses training-only preprocessing, a fixed checkpoint rule, and exact expected/uncensored latent factual metrics. Corrected checkpoints/results go to `models/semisynthetic_v2` and `reports/semisynthetic_sweep_v2`; retraining is required. See `reports/p1_fixes_2026-09-23.md` for migration instructions.

> Comparator qualification: at historical bestCI, DynaSurv's mean pair PEHE is worse than naive per-arm Kaplan–Meier in **9/10 cells** (reference 1.761 vs 1.705 months), although reference policy regret is lower (0.336 vs 0.516 months). Only heterogeneity 2 favors DynaSurv on PEHE (2.643 vs 2.773). This supports a possible ranking benefit, not superior effect estimation. The complete historical comparison is in `reports/semisynthetic_legacy_audit_comparison/comparator_summary.csv`; the new aggregator reports all comparators explicitly.

Historical status as of 2026-09-22. Code: `src/CausalSurv/semisynthetic/`, `scripts/semisynthetic/`,
`configs/semisynthetic/`, `slurm/sweep.sh`. The full 30-run
sweep has now run on the cluster and been synced back (`make sync`); results below are
aggregated across all 10 cells x 3 replicates by `scripts/semisynthetic/aggregate.py`, which
writes the raw concatenated tables and summaries to `reports/semisynthetic_sweep/`. The LaTeX
version of the design choices and the results is `latex/appendix_semisynthetic.tex` (section
"Sweep results").

## 1. Why this exists

On real data each patient shows one factual outcome per line, so DynaSurv's counterfactual
curves, treatment effects and recommendations cannot be validated. The benchmark keeps the real
covariates and treatment history of the HR+/HER2- cohort and *simulates* the current-line
treatment, the potential survival time under every arm, and the censoring. The true
S_a(t), RMST and best arm of every patient are then known in closed form.

The June 2026 version of this benchmark was lost (no code in the tree, history or worktrees; only
stale outputs in `data/synthetic/`). This is a rebuild from scratch with a different design (four
arms, prefix expansion, ≥ 2018 cohort). The paper's Section "Counterfactual Validation" still
describes the lost version (three arms, hidden-confounding results for Cox and DeepSurv) and needs
rewriting once this one has results.

The one lesson kept from the lost run: raising *observed* confounding leaves g-formula baselines
(Cox, DeepSurv) unbiased and only adds variance; only *hidden* confounding produced a monotone
RMST bias. The sweep is therefore built around a hidden-confounding axis, and bias is reported
separately from error.

## 2. Pipeline

```
real V2 cohort ──► drivers z ──► hidden u ──► assignment ──► potential outcomes ──► censoring
   (drivers.py)   (drivers.py)  (drivers.py)  (assignment.py)    (outcome.py)      (censoring.py)
                                                       │
                          expand.py (prefix expansion) ◄┘  ──► generate.py writes
                                                              expanded V2 parquet, static parquet,
                                                              truth.parquet, manifest.json
   datamodule.py ──► TrainDynasurvCausal.py (--datamodule semisynthetic) ──► evaluate.py
```

| Stage | File | Role |
|---|---|---|
| Config | `semisynthetic/config.py`, `configs/semisynthetic/dgp.toml` | Strict typed DGP config; rejects unknown keys and causal-role violations |
| Drivers | `drivers.py` | Cohort loading, 16 drivers, hidden confounder |
| Assignment | `assignment.py` | Softmax policy with calibrated intercepts |
| Outcome | `outcome.py` | Weibull potential outcomes, closed-form S and RMST, fit to real KM |
| Censoring | `censoring.py` | Administrative cutoff plus calibrated exponential dropout |
| Expansion | `expand.py` | One sample per (patient, line) with real history |
| Generation | `generate.py`, `scripts/semisynthetic/generate.py` | Orchestration, seeding, artefacts |
| Data loading | `semisynthetic/datamodule.py` | Mask on the last line, patient-grouped splits |
| Training | `scripts/TrainDynasurvCausal.py`, `scripts/semisynthetic/train.py` | Same script as real data, different datamodule |
| Evaluation | `predictors.py`, `evaluate.py`, `scripts/semisynthetic/evaluate.py` | Metrics against the truth |
| Sweep | `slurm/sweep.sh` | Cluster array job |

## 3. Design choices

Each choice states what was decided and why. Numbers are from the reference cell.

### 3.1 Cohort

* HR+/HER2- V2 file, patients whose **first line starts in 2018 or later**, lines 1–4.
  Built exactly as `ESMEOnlineDataModuleCV._load_data` builds it (patient-level entry-year filter,
  inner join with the static file), so simulated rows line up one-to-one with what the model trains
  on. 6,191 patients; 13,116 (patient, line) rows: 6,191 / 3,401 / 2,169 / 1,355 per line.
* Why ≥ 2018: it is the identifiability window of the real pipeline (arms available throughout).
* One exact duplicate (patient, line) row in V2 is dropped, because in the expansion it would
  become its own sample and hand the second copy a fake "previous line".
* Administrative cutoff 2024-02-21, the latest `line_start_date` in V2.

### 3.2 Arms

Four arms: **ET alone, ET+ANTI-CDK wo CT, MONOCT std alone, POLYCT alone.** CT+ANTI-ANGIO was
dropped (21 / 7 / 2 / 2 rows per line in the ≥ 2018 cohort: no overlap). The other categories
(`CT+ANTI-ANGIO`, `CT+ANTI-HER2`, `CT+IT`, `CT+TT`, `ET+TT`, `NO TREATMENT`, `OTHER`) stay in
the data only as *history* and are excluded from the recommendable set in the training config.
Arm indices always come from `sorted(arms)`, which matches the datamodule's alphabetical one-hot.

### 3.3 Real history, simulated present, no outcome feedback

Each patient keeps real covariates, real past treatments and real line structure. Only the
treatment at the line being scored and its outcome are simulated. The simulated outcome never
feeds back into later lines (previous-line length, gap, line number stay real). This keeps the
covariate distribution real and makes every line an independent problem given its history; it
also means the benchmark does not test the model's handling of treatment-outcome feedback across
lines.

### 3.4 Prefix expansion

A patient with L lines becomes **L samples**: sample j holds the real rows for lines < j plus row
j with the simulated arm and outcome. Sample id = `orig_usubjid * 10 + j`. 13,116 samples,
24,920 rows.

* Why: `forward()` steps lines unidirectionally and `h_t` does not depend on the current arm, so
  scoring only the last line of each sample needs no model or loss change. The alternative
  (simulate all lines jointly per patient) would need outcome feedback and a joint model.
* The datamodule masks every row except `lineid == prefix_line`. History rows have `Y = 0`.
* Current-line columns that would sit next to a simulated value are dropped: all `T_*` drug flags
  except `T_treatment_category`, all `Y_*` except the two targets, `line_end_date`, `death_date`.
  (`X_onset_to_progression`, the leaked current-line duration, is already absent from V2 and is
  excluded by the training config anyway.)
* The static file is re-keyed to the sample ids (one copy per sample).

### 3.5 Drivers

Sixteen drivers, z-scored where continuous (mpps, log previous-line length, cumulative new sites,
age, calendar), 0/1 flags otherwise. The role of each is enforced at config load time: an
instrument with an outcome coefficient, or a prognostic factor with an assignment coefficient,
raises.

| Role | Drivers | In |
|---|---|---|
| Confounder (11) | liver mets > 0, visceral (pulmonary or pleural) > 0, bone-only (bone > 0 and none of liver, visceral, brain), `X_mpps` (imputed ECOG/WHO performance status), log1p `X_time_between_onsets`, previous-line class (CDK / ET / CT, one-hot of the *real* previous arm), cumulative new sites, age, menopause | assignment and outcome |
| Instrument (1) | calendar months since 2018-01-01 (same formula as the datamodule) | assignment only |
| Prognostic (4) | brain mets > 0, lobular histology, BRCA1/2, progression of old site | outcome only |
| Hidden (1) | latent u | assignment and outcome, never written to the model input |

* **Static features never reach the model** (`_init_lstm_states` returns zeros). Age and
  menopause are therefore copied from the static file into dynamic `X_` columns; otherwise they
  would be unplanned hidden confounders.
* The model still sees all 73 `X_` columns (74 with calendar). Only 16 matter; the rest are real
  covariates the model must learn to ignore.
* `X_buffer_time` is deliberately not a driver.
* The hidden confounder is `u = ρ·z(liver) + sqrt(1−ρ²)·ε`, ρ = 0.3, ε ~ N(0,1): correlated with a
  real covariate, so it is not independent noise.

### 3.5.1 Coefficients (from `dgp.toml`)

Assignment coefficients B (per arm, per driver; empty = 0):

| Arm | Coefficients |
|---|---|
| ET alone | age 0.4, prev_et −0.5, prev_cdk −0.3 |
| ET+ANTI-CDK wo CT | bone_only 0.5, log_prev_line 0.4, prev_cdk −1.0, liver −0.3, visceral −0.2, mpps −0.3, menopause 0.3, calendar 0.3, cum_new_sites −0.1 |
| MONOCT std alone | liver 0.5, visceral 0.3, bone_only −0.4, mpps 0.3, log_prev_line −0.4, prev_ct 0.3, prev_cdk 0.4, age 0.3, cum_new_sites 0.2, calendar −0.1 |
| POLYCT alone | liver 0.9, visceral 0.5, bone_only −0.6, mpps −0.2, log_prev_line −0.5, prev_cdk 0.3, age −0.4, menopause −0.3, cum_new_sites 0.3, calendar −0.2 |

Prognostic f: liver 0.45, visceral 0.25, bone_only −0.30, mpps 0.30, log_prev_line −0.25,
prev_cdk 0.20, cum_new_sites 0.15, age 0.10, menopause 0.05, brain 0.40, lobular 0.10, brca 0.15,
old_site_progression 0.15.

Arm effects τ: ET alone 0, ET+ANTI-CDK −0.20, MONOCT +0.10, POLYCT 0 (log-hazard; negative =
better).

Interactions g_a (who benefits from what):

| Arm | Coefficients |
|---|---|
| ET alone | bone_only −0.15, prev_et 0.30, liver 0.30 |
| ET+ANTI-CDK wo CT | bone_only −0.30, log_prev_line −0.20, prev_cdk 0.40, liver 0.25 |
| MONOCT std alone | mpps −0.10, age −0.10, liver −0.15 |
| POLYCT alone | liver −0.35, visceral −0.25, mpps 0.15, age 0.15 |

Hidden confounder: assignment loadings POLYCT 1.0, MONOCT 0.5, ET+CDK −0.5 (ET alone 0); outcome
coefficient 0.4 (fixed).

All coefficients are hand-set to be clinically plausible in sign (visceral crisis → combination
chemotherapy, long previous line → endocrine-sensitive, CDK rarely repeated) and moderate in
size. They are **not fitted to anything**; they set the difficulty of the benchmark, not its
realism.

τ for ET+ANTI-CDK was −0.35 at first. That made CDK the best arm for 63–74% of patients (by line)
and ET alone best for 2–5%, so "always CDK" scored ~65% on best-arm hit rate. It was changed to
−0.20, which on line 2 lowers CDK's best-arm share from 63% to 48% (ET alone 3% → 12%), so the
recommendation metric can discriminate.

### 3.6 Assignment

```
pi_a = softmax_a( alpha[line, a] + gamma * B_a · z + strength * c_a * u )
```

* `alpha` is **calibrated per line by iterative proportional fitting** so that the marginal
  simulated arm mix equals the real within-four-arms mix, to 1e-6, at every gamma and strength.
  A sweep therefore changes *who* gets which arm, never *how many*. The mean runs over all rows of
  the line (every row is given a simulated arm) while the target is the real mix among the four
  arms. `alpha` is centred per line (softmax shift invariance).
* Target mix (ET, CDK, MONOCT, POLYCT): line 1 .183/.628/.090/.100; line 2 .148/.412/.330/.109;
  line 3 .049/.107/.611/.233; line 4 .032/.031/.580/.357.
* The arm is drawn by inverse CDF with **one uniform per row**, so draws are identical wherever
  the propensities are identical.
* True propensities are stored (`truth.parquet`).

### 3.7 Potential outcomes

```
S_a(t) = exp( -(t/lambda_line)^k_line * exp(eta_a) )
eta_a  = f(z) + tau_a + h * g_a(z) + c_out * u
```

i.e. each arm's time is Weibull(k, lambda·exp(−eta/k)); eta > 0 is a higher hazard.

* f and g_a are **centred within each line**, so (k, lambda) are the baseline of the average
  patient and can be fitted to the real Kaplan-Meier.
* **Fitting (k, lambda) per line:** least squares between the mixture of the factual arms'
  survival curves and the real per-line KM, over the range where at least 5% of the line is still
  at risk. Result: k ≈ 1.27–1.33, lambda ≈ 53 / 27 / 22 / 18 months. Fitted medians
  48.5 / 22.6 / 16.3 / 12.7 vs real KM 47.7 / 22.8 / 16.5 / 12.4 months.
* **Common random numbers across arms:** one uniform per row is shared by all four arms (a monotone
  coupling), so arms differ only through eta. Consequence: the truth is one particular joint
  distribution of potential outcomes; a different coupling would give the same marginals and
  different individual-level effects.
* Truth is closed form: `S_a(t)` above, and `RMST_a(tau) = s·Γ(1+1/k)·P(1/k, (tau/s)^k)` with
  `s = lambda·exp(−eta/k)` and P the regularised incomplete gamma. Checked against Monte Carlo:
  bias −0.002 months, z-scores sd 1.01.
* Time unit: months from line onset to global death (`Y_onset_to_death`, `Y_global_death_status`),
  the targets the real pipeline uses.

### 3.8 Censoring

```
C = min( administrative, dropout )
administrative = (2024-02-21 - real line_start_date) / 30.44     months
dropout        ~ Exponential(rate_line), independent of T and of the arm
observed time  = min(T, C), event = 1{T <= C}
```

* Administrative censoring is a fact of the real cohort and is kept per row.
* Dropout is added only where the cutoff alone leaves the simulated event rate above the real
  one; `rate_line` is bisected (60 steps on the log-rate) so the rates agree, using fixed Exp(1)
  draws so the event rate is a deterministic decreasing function of the rate. Rate 0 if the cutoff
  alone already gives no more events than the target.
* Targets are the ≥ 2018 event rates 0.442 / 0.547 / 0.578 / 0.596, **not** the full 2008+ cohort
  (0.73–0.85). Cutoff-only rates were 0.465 / 0.587 / 0.639 / 0.679, so dropout is on for all
  lines: 0.0024 / 0.0041 / 0.0073 / 0.0103 per month.
* Independent dropout does not establish marginal independent censoring. Administrative
  follow-up depends on calendar date, which also affects assignment and patient mix. Training
  censoring weights need not transfer to the temporal test cohort; protocol v2 uses exact
  expected and uncensored latent factual scores instead.

### 3.9 Seeds and common random numbers

`SeedSequence([dgp.seed, replicate]).spawn(4)` gives independent streams for the hidden
confounder, assignment, outcome and censoring. A sweep over gamma, strength or heterogeneity
reuses the same u, arm uniforms, latent uniforms and dropout draws, so levels differ only by the
knob. Re-running the same knobs is bit-identical (checked).

### 3.10 Artefacts per replicate

`data/semisynthetic/{axis}/{level}/rep{r}/`:

* the expanded dynamic parquet and re-keyed static parquet, under the **real V2 file names**, so
  the directory can be given to the datamodule as `data_dir`;
* `truth.parquet` (41 columns): sample and original ids, line, assigned arm, hidden u, Weibull k
  and lambda, latent time under every arm, both censoring times, observed time and event,
  propensity and eta per arm, all drivers;
* `manifest.json`: every parameter, calibrated alpha, per-line arm counts, ESS/n, fitted Weibull
  parameters and medians, dropout rates, achieved and target event rates, git hash.

The hidden u appears **only** in `truth.parquet`.

### 3.11 Data loading (`SemiSyntheticDataModule`)

* Mask keeps only the sample's last line; valid-arm counts are recomputed on those rows.
* Every non-temporal split (random holdout, CV folds, early-stopping split) **groups by
  `orig_usubjid`**, because the samples of one patient share history and would leak across a
  train/validation boundary. A temporal split already groups them (shared entry year).
* CV folds are taken inside the training partition only. (The real-data datamodule's CV mode folds
  over the whole dataset including the holdout; that was left unchanged.)
* The training loader shuffles with a seeded generator (the base loader does not).
* The base class got two behaviour-preserving hooks (`_split_holdout`, `_cv_split`), verified
  against the pre-edit datamodule on real data (identical holdout and validation indices).

### 3.12 Training configuration (`configs/semisynthetic/config.toml`)

Same schema as the real config. Differences: every arm outside the four is excluded from the
recommendable set; support thresholds lowered to 30 / 10 / 30 (min samples / events / follow-up),
because the four arms thin out at lines 3–4; early stopping on `val_loss`, patience 20, max 150
epochs; calendar feature on (it is the instrument); temporal split at 2021 (holdout 26% of
samples). The model config is the real-data `best_config.json`, which comes from an Optuna study
that was inflated by the leaked covariate and has not been re-tuned for this data.

### 3.13 Evaluation

All quantities are computed at each sample's own (last) line and that line's horizon from
`[eval] horizon_times` = 24 / 18 / 12 / 12 months, on the temporal holdout, on a 0.1-month grid.
Arms outside the model's line-level support are excluded from every table.

| Table | Content |
|---|---|
| curve | RMSE of S(t) to the horizon and RMST error (MAE and bias), split into the factual arm and the counterfactual arms |
| effect | Per arm pair: PEHE, bias of the average effect, sign agreement |
| policy | Value (true RMST of the arm chosen), regret against the best arm, hit rate, abstention rate |
| factual | Protocol v2: exact expected Brier, uncensored latent-outcome Brier and concordance at the specified horizon; historical tables below used training-cohort IPCW and require replacement |

Policies: DynaSurv with the patient-level support mask (`supported`, uses the recommender's
propensity-overlap mask; abstains to the assigned arm), DynaSurv with line-level support only
(`line_only`), and references: assigned arm (`factual`), best single arm per line by true mean RMST
(`best_constant`, an upper bound for any constant policy), uniform random.

Reference predictors: **oracle** (true curves; floor, must score 0) and **naive per-arm KM**
(training-set Kaplan-Meier per line and arm, no covariates; the bar a covariate model must clear).

### 3.14 Sweep

One knob at a time, others at the file's values (gamma 1, strength 0, heterogeneity 1):

| Axis | Levels | Note |
|---|---|---|
| gamma | 0, 0.25, 0.5, 1.0, 1.5 | 1.0 is the reference cell |
| hidden strength | 0.5, 1.0 | 0 is `gamma/1.0` |
| heterogeneity | 0, 0.5, 2.0 | 1 is `gamma/1.0` |

10 cells × 3 replicates = 30 array tasks; replicate r uses data seed r and training seed r.
Levels gamma 2 and 4 and strength 2 were dropped: effective sample size falls to ≈ 0.01 (gamma ≥ 2)
and 0.07 (strength 2), which measures extrapolation, not confounding.

## 4. Verification

| Check | Result |
|---|---|
| Alpha calibration | mean pi per line equals the target to < 1e-6 for gamma ∈ {0, 0.5, 1, 2, 4} × strength ∈ {0, 2}; sampled counts pass chi-square |
| gamma = 0, strength = 0 | pi constant within each line |
| Recovery of B | pooled multinomial refit with line dummies: correlation 0.976, sign agreement 100% (line 1 alone 0.58: previous-line drivers are constant there) |
| Weibull fit | medians within 2% of real KM; S(t) within ~0.04 at 6/12/24/36 months |
| Closed-form vs Monte Carlo | RMST(24) bias −0.002, z sd 1.01 |
| h = 0, τ = 0 | eta identical across arms (true PEHE 0) |
| Censoring | achieved event rates equal the targets; observed KM within ~0.02 of the real one |
| Expansion | endpoint rows per line equal real patients per line; history rows equal real arms/covariates; endpoint Y equals truth; u absent from model files |
| Reproducibility | identical rerun identical; gamma change keeps u and moves 13% of arms; new replicate gives new u |
| Datamodule | one unmasked line per sample; zero patient overlap across temporal, random and CV splits |
| Refactor hooks | identical holdout and validation indices vs pre-edit code on real data |
| Evaluator | oracle: error 0, PEHE ≤ 1e-5, regret 0, quadrature residual ≤ 1e-4; naive-KM factual bias matches an independent computation exactly (0.6306, n = 365) |

## 5. Results: the full sweep (30 runs, 10 cells x 3 replicates)

Manifests for all 30 runs are now synced (`rsync` of `data/semisynthetic/*/*/rep*/manifest.json`);
the reference-cell training diagnostic still stands from the single-cell smoke test (val loss
8.7 → 1.685 at epoch 8, drifting to ≈1.73 by early stop at epoch 28).

**Overlap (effective sample size / n per line, mean over 3 replicates)** — this is the mechanism
behind the gamma/strength results below: as either axis strengthens, overlap collapses, and PEHE
degrades with it.

| Axis | Level | Line 1 | Line 2 | Line 3 | Line 4 |
|---|---|---|---|---|---|
| gamma | 0 | 0.566 | 0.750 | 0.440 | 0.239 |
| gamma | 0.25 | 0.539 | 0.709 | 0.422 | 0.217 |
| gamma | 0.5 | 0.462 | 0.586 | 0.364 | 0.184 |
| gamma | 1 (ref) | 0.251 | 0.282 | 0.198 | 0.086 |
| gamma | 1.5 | 0.087 | 0.154 | 0.085 | 0.097 |
| strength | 0.5 | 0.197 | 0.275 | 0.140 | 0.061 |
| strength | 1.0 | 0.156 | 0.199 | 0.109 | 0.098 |
| heterogeneity | any | 0.251 | 0.282 | 0.198 | 0.086 |

Heterogeneity is identical to the gamma=1 reference at every level, as it should be: `h` only
scales the outcome model, never the assignment mechanism, so it does not touch overlap. Achieved
censoring event rates matched their per-line targets to within 0.0007 in every one of the 30
runs (no cell needed re-calibration), confirming the bisection-based censoring calibration
(`src/CausalSurv/semisynthetic/censoring.py`) generalizes across the whole sweep, not just the
reference cell.

**Checkpoint kind, pooled over all 30 runs and all 4 kinds** (`n`-weighted means; PEHE and
`|`bias`|` in months, regret in months, oracle = 0):

| Checkpoint | PEHE | mean \|ATE bias\| | Regret | Hit rate | Factual C |
|---|---|---|---|---|---|
| val_loss (real-data default) | 2.37 | 1.51 | 0.42 | 0.62 | 0.666 |
| bestCI | 1.95 | 1.14 | 0.36 | 0.64 | 0.675 |
| bestCALIB | 2.43 | 1.35 | 0.37 | 0.58 | 0.665 |
| final_epoch | 2.23 | 1.16 | 0.40 | 0.52 | 0.663 |

`bestCI` wins on PEHE, regret and factual C-index; `val_loss` is worst on every column except
hit rate. This confirms, on 30 runs instead of 1, that validation loss on the factual likelihood
does not track counterfactual quality here. The tables below use `bestCI`.

**Sweep axes, mean ± std over 3 replicates** (PEHE and `|`bias`|` in months, pooled over lines
and arm pairs; regret in months against the `line_only` policy; `h_r` = random-policy regret,
an upper bound on that cell's headroom):

| Axis | Level | PEHE | \|bias\| | Regret | Hit rate | h_r |
|---|---|---|---|---|---|---|
| gamma | 0 | 1.27 ± 0.15 | 0.34 ± 0.15 | 0.41 ± 0.03 | 0.57 ± 0.05 | 0.77 |
| gamma | 0.25 | 1.45 ± 0.28 | 0.70 ± 0.40 | 0.45 ± 0.02 | 0.56 ± 0.02 | 0.78 |
| gamma | 0.5 | 1.61 ± 0.19 | 0.85 ± 0.23 | 0.39 ± 0.04 | 0.59 ± 0.04 | 0.79 |
| gamma | 1 (ref) | 1.76 ± 0.42 | 0.87 ± 0.33 | 0.34 ± 0.04 | 0.64 ± 0.04 | 0.80 |
| gamma | 1.5 | 2.33 ± 1.10 | 1.49 ± 1.03 | 0.40 ± 0.05 | 0.62 ± 0.02 | 0.81 |
| strength | 0.5 | 2.25 ± 0.32 | 1.49 ± 0.24 | 0.44 ± 0.04 | 0.59 ± 0.02 | 0.80 |
| strength | 1.0 | 2.70 ± 0.15 | 2.14 ± 0.16 | 0.48 ± 0.01 | 0.57 ± 0.01 | 0.80 |
| heterogeneity | 0 | 2.08 ± 0.75 | 1.28 ± 0.34 | 0.03 ± 0.03 | 0.95 ± 0.03 | 0.46 |
| heterogeneity | 0.5 | 1.46 ± 0.29 | 0.94 ± 0.45 | 0.15 ± 0.08 | 0.67 ± 0.12 | 0.50 |
| heterogeneity | 2.0 | 2.64 ± 0.53 | 1.35 ± 0.49 | 0.52 ± 0.07 | 0.62 ± 0.01 | 1.46 |

Reading:
- **gamma (observed-covariate confounding):** PEHE and average-effect bias roughly double from
  gamma=0 to gamma=1.5, but the model's own regret stays flat within noise (0.34–0.45) while the
  random-policy headroom barely moves (0.77 → 0.81). Ranking arms survives better than sizing the
  effect does as overlap collapses — consistent with variance rather than bias, though 3
  replicates cannot rule out a small true bias on this axis.
- **strength (hidden confounder into assignment):** the sharper effect. Bias more than doubles
  from the strength=0 reference (0.87) to strength=1 (2.14), and regret rises ~40% (0.34 → 0.48)
  — the pattern expected of a confounder the model never sees.
- **heterogeneity:** mostly changes the size of the task. At h=0 every line has one best arm, so
  both the model and a constant policy score near-zero regret; as h grows to 2 the headroom
  triples (0.46 → 1.46) and the model's regret grows with it (0.03 → 0.52) while staying well
  under the random policy's ceiling; hit rate falls from 0.95 to 0.62 as there are more genuinely
  different arms to pick between.

**What this does not show:** no Cox/DeepSurv baseline is in this pipeline yet, so the sweep
cannot reproduce the lost benchmark's observed-vs-hidden-confounding bias comparison for those
methods — only DynaSurv itself is scored, against the oracle and the covariate-free KM. Raw and
summary CSVs: `reports/semisynthetic_sweep/`.

## 6. Limitations and open decisions

1. **3 replicates per cell.** Enough for a mean ± std, not for a significance test; the gamma
   axis in particular has a std comparable to its trend (Section 5).
2. **Checkpoint kinds were compared after seeing the truth**, so `bestCI`'s win is a selection
   effect over 4 candidates, now measured on 30 runs instead of 1 but still not held out.
3. **Open decision: early-stopping monitor.** The synthetic config stops on `val_loss`; the real
   config stops on `val/calib_gap_abs_mean`. The first run stopped one epoch after its
   best-calibration checkpoint. Mirroring the real pipeline would need a config change and a
   retrain. Not done.
4. **Coefficients are hand-set**, not estimated; results describe this DGP's difficulty, not real
   effect sizes.
5. **Weibull with one shape per line**, shared coupling across arms, calendar-dependent administrative censoring,
   no outcome feedback across lines, no time-varying treatment within a line.
6. **Only 16 covariates matter.** The model sees 73–74 real columns; the rest are non-causal for
   the outcome by construction, which is easier than reality in one way (no unmodelled real
   signal) and harder in another (many nuisance columns).
7. **Thin arms at late lines** (ET alone, CDK at line 4): their counterfactuals are extrapolation;
   metrics are reported per arm pair with support and drop unsupported arms.
8. **Calendar is an instrument and the holdout is late-entry**, a deliberate extrapolation shift.
9. **Standardiser is fitted on the frame including the holdout** (existing caveat of the datamodule).
10. **Static features do not reach the model** (existing model property); the DGP works around it
    by copying age and menopause into dynamic columns.
11. The tuned model config predates this data and is not re-tuned; the synthetic training
    schedule was chosen by hand.

## 7. How to run

From `scripts/`:

```bash
# one cell, one replicate
python semisynthetic/generate.py --out ../data/semisynthetic/gamma/1.0/rep0 --replicate 0 --gamma 1.0
python semisynthetic/train.py    --axis gamma --level 1.0 --rep 0 --seed 0     # ~7 s/epoch locally
python semisynthetic/evaluate.py --axis gamma --level 1.0 --rep 0 --seed 0 --kind bestCALIB
```

Knobs of `generate.py`: `--gamma`, `--strength`, `--heterogeneity`. Outputs land in
`models/semisynthetic/{axis}/{level}/rep{r}/seed_{s}/eval_{kind}/{curve,effect,policy,factual}.csv`.

Cluster: `sbatch slurm/sweep.sh` (30 tasks; `--array=1-30%5` to throttle; `DRY_RUN=1
SLURM_ARRAY_TASK_ID=7 bash slurm/sweep.sh` prints a task without running it). Then `make sync`
pulls `models/`; `data/` (including `manifest.json` and `truth.parquet`) stays on the cluster.
The wall-clock time per task on the cluster is untimed; the script asks for 4 h.

## 8. Files touched

New: `src/CausalSurv/semisynthetic/{assignment,outcome,censoring,expand,generate,datamodule,predictors,evaluate}.py`,
`scripts/semisynthetic/{generate,train,evaluate,aggregate}.py`, `configs/semisynthetic/config.toml`,
`slurm/sweep.sh`, `latex/appendix_semisynthetic.tex`, `reports/semisynthetic_sweep/` (aggregated
sweep CSVs). Committed earlier (4a88a63): `semisynthetic/{__init__,config,drivers}.py`,
`configs/semisynthetic/dgp.toml`. Modified: `configs/semisynthetic/dgp.toml` (τ for ET+CDK),
`scripts/TrainDynasurvCausal.py` (datamodule, run dir, W&B project arguments),
`src/CausalSurv/data/datamodule_cv.py` (two split hooks), `latex/second_draft.tex`
(Counterfactual Validation section: real 30-run sweep results, four arms instead of three).
