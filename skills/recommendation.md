# Ensemble recommendation strategy

Last checked: **7 October 2026**. The implementation contract below describes
the active pipeline. The later 40-model figure walkthrough documents a
**historical September 2026 report**, not a newly validated ensemble.

## Current implementation and execution

- [TreatmentRecommender](../src/CausalSurv/recommendation/recommender.py) computes
  all-arm survival/RMST and applies global and patient-specific support.
- [ensemble.py](../src/CausalSurv/recommendation/ensemble.py) selects checkpoints,
  checks members, combines predictions, and produces four decision statuses.
- [pipeline.py](../src/CausalSurv/recommendation/pipeline.py) reconstructs the
  datamodule and checks it against checkpoint manifests.
- [RecommendEnsemble.py](../scripts/RecommendEnsemble.py) writes recommendations;
  [HorizonRobustness.py](../scripts/HorizonRobustness.py) rescales the RMST horizon
  while keeping curves and support fixed. It does not revalidate support at each
  alternative horizon.

Run the CLIs from `scripts/`. Both load `../configs/config.toml` and
`../configs/hpo_v3/best_config.json` by default. The latter is absent locally at
this review. `--config`, `--model-config`, `--runs`, `--kind` and the decision
threshold flags override defaults. Reports go to
`reports/recommendations/{subtype}_{n_lines}lines/{timestamp}_{kind}_M{members}/`;
the main CLI writes `recommendations.parquet` and `summary.json`, while the
horizon CLI writes `horizon_sweep.parquet` and `horizon_summary.json` in a
directory ending in `_horizons`.

The current config uses `checkpoint_kind="val_loss"`, `min_members=2`,
`leader_rule="lcb"`, `pessimism_c=1`, `margin_months=1`, and `p_best_min=0.7`.
The checkpoint rule and thresholds must be fixed using development data before
reporting test recommendations. The CLI scores the datamodule's reserved test
partition; it does not estimate observed benefits from following the policy.

Support counts and outcome/horizon thresholds use training data. The separate
per-line propensity classifier estimates assignment across **all observed
classes**, including excluded arms; it does not renormalize over eligible arms.
Missing assignment models yield no verified patient-level support. See
[eligibility filters](../recommendation_filters.md) for the exact rules.

`load_member` requires static-enabled weights, a protocol-v2 manifest, an
all-observed-class assignment model and persisted eligibility. Assembly checks
the current datamodule manifest against each member and rejects inconsistent
dimensions, grids, architectures, eligibility, propensity classes/thresholds,
MLP normalization, or calibration status. BatchNorm and LayerNorm members cannot
be combined in one ensemble. Training-setting differences are warnings. Missing or
incompatible runs abort unless `--skip-incompatible` is supplied; the retained
ensemble must still meet `min_members`.

The single-model `TreatmentRecommender.recommend` returns the highest-RMST
supported arm or `-1`; it has no ensemble confidence test. The ensemble emits
`confident`, `undecided`, `only_supported_option`, or `no_support`. Exactly one
eligible arm never produces a confident comparison. The mathematical rule below
matches this distinction.

Only one active local model run was found, fewer than the default two-member
requirement. No fresh ensemble was assembled in this review. The historical
40-member reports predate the corrected split/support requirements and do not
establish current performance.

## Algorithm and historical figure walkthrough

This document tracks the current recommendation pipeline and its evolution. It describes how the ensemble selects a treatment leader, evaluates agreement and separation from alternatives, and decides whether to issue a recommendation. It also explains how `data_analysis/recommendation_mix.py` displays those decisions.

For each patient and treatment line, the recommendation algorithm takes as input the restricted mean survival time (RMST) predicted by K trained models for each eligible treatment. It combines these predictions to select a single treatment leader. A recommendation is issued when that leader has sufficient model agreement and a sufficiently large conservative advantage over every eligible alternative.

## Mathematical definition of the recommendation algorithm

Consider one patient at one treatment line. Let $R_{k,a}$ denote the RMST predicted by model $k\in\{1,\ldots,K\}$ under treatment $a$, using the same time horizon for all models and treatments at this line. Let $\mathcal A_k$ be the treatments supported by model $k$. The algorithm compares treatments in their common eligible set:

$$
\mathcal A = \bigcap_{k=1}^{K}\mathcal A_k.
$$

For $K\geq2$, define the mean and sample standard deviation of any K model-level values $z_1,\ldots,z_K$ as:

$$
\bar z=\frac{1}{K}\sum_{k=1}^{K}z_k,
\qquad
s(z)=\sqrt{\frac{1}{K-1}\sum_{k=1}^{K}(z_k-\bar z)^2}.
$$

**Leader selection.** For each eligible treatment, compute $\mu_a=\overline{R_{\cdot,a}}$ and $\sigma_a=s(R_{\cdot,a})$. With the conservative scoring rule, the ensemble leader is:

$$
L=\arg\max_{a\in\mathcal A}(\mu_a-c\sigma_a),
$$

where $c\geq0$ controls the penalty for variation across models. This selects one fixed leader for the patient and line.

**Model agreement.** Each model votes for its own highest-RMST treatment within the same eligible set. The leader vote share is:

$$
v_k=\arg\max_{a\in\mathcal A}R_{k,a},
\qquad
p_L=\frac{1}{K}\sum_{k=1}^{K}\mathbf 1\{v_k=L\}.
$$

Here, $\mathbf 1\{v_k=L\}$ equals 1 when model $k$ votes for the ensemble leader and 0 otherwise. The votes measure how often the selected leader ranks first; they do not determine the leader. The code resolves tied maxima using the lowest treatment index.

**Comparison with alternatives.** For every eligible rival $a\ne L$, compute the paired differences and their conservative gap:

$$
d_{k,a}=R_{k,L}-R_{k,a},
\qquad
g_a=\bar d_a-c\,s(d_{\cdot,a}).
$$

The leader and rival are fixed across the K models. The figure displays the smallest rival gap:

$$
g=\min_{a\in\mathcal A\setminus\{L\}}g_a.
$$

The rival attaining this minimum is the figure's “runner-up”: the alternative with the weakest conservative separation from the leader.

**Decision rule.** Let $\delta\geq0$ be the required RMST advantage in months and $q\in[0,1]$ the required vote share. A confident recommendation is issued exactly when:

$$
|\mathcal A|\geq2,
\qquad p_L\geq q,
\qquad g\geq\delta,
\qquad g>0.
$$

For a positive margin, $g\geq\delta$ already implies $g>0$. The explicit positivity condition matches the code when $\delta=0$: a zero gap does not separate a rival. The implementation allows a small floating-point tolerance in the vote comparison.

The recommended treatment is L when these conditions hold. Otherwise, the result is **undecided** when at least two treatments are eligible, **only supported option** when exactly one is eligible, or **no support** when none is eligible. For one eligible treatment, the code records $g=+\infty$ as a no-rival sentinel; for no eligible treatments, the gap is undefined. Neither case produces a confident recommendation.

## Overall meaning of the figure

The figure has one column per treatment line and two rows. The top row compares the treatments prescribed by clinicians with the leaders selected by the recommendation algorithm, distinguishing confident recommendations from undecided cases. The bottom row shows, for each patient with a supported treatment, the leader's conservative RMST advantage over its closest eligible rival and the fraction of the K models ranking the leader first.

The figure describes predictions and decisions. A difference from clinicians' choices does not itself demonstrate better outcomes, and a confident recommendation does not establish that the prediction is correct.

## Historical example: from 40 models to a recommendation

All calculations below concern one patient at one treatment line. Each of the 40 models predicts an RMST for each treatment. RMST is the area under the predicted survival curve up to the chosen time horizon, expressed in months.

### 1. Determine the eligible treatments

A treatment must pass the algorithm's data-support rules:

- It has enough training examples at this treatment line, with event and follow-up requirements when configured.
- Its category is not explicitly excluded from recommendation.
- Its estimated treatment-assignment probability for this patient reaches the configured support threshold.
- All ensemble members mark it as supported; the ensemble uses the intersection of their support masks.

Eligibility is determined before ranking. It describes statistical support under the implemented rules, rather than a clinical assessment of contraindications or suitability.

An **eligible rival** is any eligible treatment other than the ensemble leader.

### 2. Select one ensemble leader

For each eligible treatment, calculate its mean RMST and standard deviation across the 40 models. In this report, the score is:

$$
s_a = \operatorname{mean}_m(R_{m,a}) - \operatorname{SD}_m(R_{m,a}).
$$

Here, $R_{m,a}$ is model $m$'s predicted RMST for treatment $a$. The treatment with the highest score becomes the **ensemble leader**.

This favours a high average RMST while penalising variation between models. More generally, the code uses mean minus $c$ times SD; this report has $c=1$ and `leader_rule="lcb"`.

There is one fixed leader for this patient–line. It is not selected separately within each model. Also, this score concerns variability in a treatment's RMST, rather than variability in a treatment-effect difference.

### 3. Calculate the leader vote share

Each model independently votes for the eligible treatment with its highest predicted RMST. These individual winners are then compared with the fixed ensemble leader:

$$
p_{leader} = \frac{\text{models ranking the ensemble leader first}}{40}.
$$

For example, if 32 models rank the leader first, its vote share is $32/40=0.80$. Each vote contributes 0.025; the report's 0.70 threshold requires at least 28 votes. Exact RMST ties are resolved by the lowest treatment index in the code.

Each vote comes from an individual model, while the vote share is an ensemble summary. The leader can differ from the most-voted treatment because the leader is chosen using mean minus SD, rather than vote counts.

Vote share measures agreement on the ranking. It is not a probability that the patient benefits, a survival probability, or a calibrated probability that the leader is truly optimal.

### 4. Compare the fixed leader with every eligible rival

For each rival $a$, calculate a paired difference within each model:

$$
d_{m,a} = R_{m,leader} - R_{m,a}.
$$

This creates **one distribution of 40 differences per rival**. The leader and that rival remain fixed across those 40 comparisons. The calculation does not compare each model's own winner with its own second-best treatment.

For each rival, calculate:

$$
g_a = \operatorname{mean}_m(d_{m,a}) - c\operatorname{SD}_m(d_{m,a}).
$$

Then take the smallest adjusted gap:

$$
g = \min_{a\ne leader,\ a\ eligible} g_a.
$$

The rival producing this minimum is called the **runner-up** in the figure. It is the hardest rival to beat under this adjusted comparison, rather than necessarily the treatment with the second-highest leader score, mean RMST, or vote share.

For example, if the adjusted gaps against three rivals are 2.5, 0.6, and 1.8 months, the plotted gap is 0.6 months.

### 5. Issue a recommendation or withhold it

For this report, a confident recommendation requires:

- At least two eligible treatments.
- A conservative gap of at least **1 month against every eligible rival**, equivalent to $g\geq1$.
- A leader vote share of at least **0.70**.

The code expresses separation through an equivalence set: eligible treatments that are not sufficiently separated from the leader remain in that set. Confidence requires the leader to be its only member, together with the vote requirement.

If several treatments are eligible but the requirements fail, the result is **undecided**. No eligible treatment gives **no support**. The current implementation labels exactly one eligible treatment **only supported option** and withholds a confident recommendation because there is no supported comparison.

## Why use the score, voting, and the gap?

| Component | Role | Limitation |
|---|---|---|
| Mean minus SD score | Chooses a candidate with high predicted RMST and less model variation | Does not guarantee it wins in most individual models |
| Leader vote share | Requires a minimum frequency of ranking that candidate first | Ignores the size of its advantage |
| Paired conservative gap | Requires a sufficiently large advantage after penalising variation in treatment differences | Is a heuristic bound, not automatically a calibrated confidence interval |

The gap check and voting can overlap: a small or unstable advantage can fail the gap check even without a voting threshold. Voting adds a separate ranking-frequency requirement; the rule alone does not demonstrate that the 70% threshold improves recommendation quality.

These checks address variation across trained ensemble members. Shared errors, confounding, omitted predictors, or a different patient population can still make a unanimous ensemble wrong.

## Reading the top row: treatment mix

The x-axis contains two bars, **clinicians** and **ensemble**. The y-axis is the share of patients observed at that line, from 0 to 100%.

Colours identify ET, ET + CDK4/6, mono-chemotherapy, and poly-chemotherapy. The clinicians bar groups other observed treatments into grey **other**.

In the ensemble bar:

- **Solid segments** represent confident recommendations, coloured by the leader.
- **Hatched segments** represent undecided cases, also coloured by the leader. These patients have a leading treatment but no issued recommendation.
- **Grey** represents no-support cases.

Each segment is $100\times$ its patient count divided by the total number of patients observed at that line. The ensemble percentages are not calculated only among confident patients. Segments at least 6% tall receive labels rounded to whole percentages.

A large coloured ensemble segment means that treatment often leads. A large hatched portion means the ensemble frequently withholds a recommendation. Comparing the bars describes aggregate distributions; it does not identify individual treatment switches or establish clinical benefit.

## Reading the bottom row: the decision plane

| Element | Meaning |
|---|---|
| One point | One patient at that treatment line, with at least one supported treatment |
| Colour | That patient's ensemble leader |
| X-axis: gap to runner-up | Minimum mean-minus-SD RMST difference against eligible rivals, in months |
| Y-axis: leader vote share | Fraction of the 40 models ranking the fixed ensemble leader first |
| Vertical dashed line | Required advantage: 1 month in this report |
| Horizontal dashed line | Required agreement: 0.70 in this report |
| Shaded upper-right region | Both numerical requirements are met; a confident recommendation also requires a supported rival |

Examples of interpretation:

- **High vote share, small gap:** models tend to agree on the ranking, but the advantage is too small or variable to meet the gap requirement.
- **Large gap, insufficient vote share:** the adjusted advantage requirement passes, but the ranking agreement requirement fails.
- **Both requirements pass:** the ensemble issues a confident recommendation if at least two treatments are eligible.

### Why can the gap be negative?

Two distinct situations can produce a negative gap:

| Mean leader–rival difference | SD of differences | Gap with $c=1$ | Interpretation |
|---:|---:|---:|---|
| −0.5 months | 0.8 months | −1.3 months | The rival has a higher mean predicted RMST |
| +0.5 months | 1.2 months | −0.7 months | The leader has a positive average advantage, but the variability penalty exceeds it |

The first situation is possible because the leader maximises mean minus SD, rather than mean RMST alone. The second is possible because a stable prediction for the leader does not ensure a stable difference against a rival. The SD of the paired difference depends on both treatments' predictions and how they vary together.

A negative plotted gap means the adjusted comparison does not establish a positive advantage over at least one eligible rival. It does not necessarily mean the leader has a lower mean RMST. Inspect `diff_mean` and `diff_std` for the rival attaining the minimum to distinguish the explanations.

Negative values carry information: below zero the adjusted advantage is not positive; between zero and one month it is positive but insufficient; at one month or above the gap requirement passes. Clamping negative gaps to zero would merge different comparisons, and hiding that region would remove patients from view.

### What does “uncertainty-adjusted” mean here?

The term refers specifically to subtracting a multiple of the SD across ensemble members from the average predicted difference. It is more precisely an adjustment for **variation across models**. It does not account for every source of uncertainty, and mean minus one SD has no automatic confidence-level interpretation.

The alternative axis wording discussed in our exchange was **“Conservative RMST advantage (months)”**. An explicit caption definition is:

> Minimum, across eligible alternatives, of the mean leader–alternative RMST difference minus its standard deviation across the 40 models.

This documents the quantity currently labelled “gap to runner-up”; no figure labels or code were changed for this recap.

## How the script builds the figure

1. Reads `recommendations.parquet` and `summary.json` from the selected report. Without a report argument, it selects the newest matching `bestCALIB` report, excluding horizon sweeps.
2. Uses `leader_table()` to reduce the per-treatment rows to one row per patient–line and reconstruct the minimum rival gap from `diff_lcb`.
3. Uses `mix_shares()` to calculate the treatment/status percentages.
4. Draws the bars with `draw_mix()` and the scatter plots with `draw_plane()`.
5. Applies the shared paper styling and saves `latex/figs/recommendation_mix.pdf` when the script is run.

The plot does not recompute model predictions or recommendation decisions.

Display details affect how to read the scatter:

- Vertical jitter of ±0.006 separates overlapping vote shares. Apparent values slightly above 1 are drawing artefacts.
- Gaps below −3 months are pinned at −3 and marked `<`.
- No-rival cases have an infinite gap and are displayed at 6.3 with `>`. This is a display position, not a measured advantage.
- Finite gaps above 6.5 months are clipped by the current axis limits.
- The y-axis starts at 0.2, so sufficiently low vote shares are outside the visible range.
- Point overlap and transparency mean visible point counts should not be treated as exact patient counts.

There is a compatibility limitation: the plotting code stacks only `confident`, `undecided`, and `no_support`. It does not include the current `only_supported_option` status. Such cases would leave the ensemble stack below 100%, while their no-rival points could appear in the shaded region despite receiving no confident recommendation. The inspected report summary lists only the three older categories.

## Historical report context and interpretation limits

The saved `20260921_175204_bestCALIB_M40` report contains 40 members and uses $c=1$, a 1-month margin, and a 0.70 vote threshold. These are legacy descriptive outputs, not results rerun with protocol-v2 checkpoints. The active CLI now defaults to `val_loss`; the plotting script separately defaults to `bestCALIB`.

| Treatment line | Patients observed | RMST horizon | Confident recommendations |
|---|---:|---:|---:|
| 1 | 1,977 | 24 months | 71.3% |
| 2 | 841 | 18 months | 44.1% |
| 3 | 438 | 12 months | 1.1% |
| 4 | 198 | 12 months | 1.0% |

These percentages describe how often the saved decision rule commits to a treatment. They do not measure accuracy. The lines involve different observed patient groups, different eligible treatment sets, and different horizons, so the confidence rates alone cannot establish why recommendations become less frequent at later lines.

## Code references

- [Figure construction](../data_analysis/recommendation_mix.py)
- [Treatment vocabulary and patient–line table](../data_analysis/recommendation_style.py)
- [Ensemble scoring, votes, differences, and decisions](../src/CausalSurv/recommendation/ensemble.py)
- [Patient-specific eligibility](../src/CausalSurv/recommendation/recommender.py)
- [Line-level eligibility](../src/CausalSurv/data/datamodule_cv.py)
- [Patient-specific propensity threshold](../src/CausalSurv/evaluation/propensity_overlap.py)
- [Historical report metadata](../reports/recommendations/HR+HER2-_4lines/20260921_175204_bestCALIB_M40/summary.json)
