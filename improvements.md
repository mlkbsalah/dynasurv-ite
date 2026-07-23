# Identifiability Improvements for DynaSurv

Recommendations arising from an empirical audit of the identifiability assumptions
stated in `latex/first_draft.tex` (§Identifiability, lines 160–196) against the real
HR+HER2− cohort.

**Data audited:** `data/model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet`
— 63,317 line-records, 19,865 patients, 11 arms in `T_treatment_category`, lines 1–4.
Propensities estimated by 5-fold out-of-fold multinomial logistic regression on the
history the model conditions on (`X_` features + previous-line treatment).

---

## The core reframe: calendar era is two problems, not one

Almost every recommendation below follows from a single distinction. Conflating these
two roles of calendar time is what makes the problem look intractable.

| | **Problem 1 — Availability** | **Problem 2 — Secular trend** |
|---|---|---|
| What it is | Which arms *existed* at time *t* | Outcomes drift over calendar time for reasons unrelated to the arm chosen |
| Causal role | Support / positivity violation | Genuine confounding |
| Does conditioning on year fix it? | **No** | **Yes** |
| Correct response | Restrict the cohort and the action set | Include calendar time in the conditioning set |

CDK4/6 inhibitors had probability **exactly zero** before their approval. No covariate
adjustment recovers a counterfactual for a 2010 patient under a drug that did not exist;
this is structural non-positivity, and the literature's prescribed responses are
restriction of the sample and modification of the target intervention — not adjustment.

> **Reference.** Petersen ML, Porter KE, Gruber S, Wang Y, van der Laan MJ.
> "Diagnosing and responding to violations in the positivity assumption."
> *Statistical Methods in Medical Research* 2012;21(1):31–54.
> [DOI](https://journals.sagepub.com/doi/abs/10.1177/0962280210386207) ·
> [PubMed](https://pubmed.ncbi.nlm.nih.gov/21030422/)

**The practical consequence.** You are not obliged to marginalise over eras. The
recommender deploys into the *current* era, for patients presenting *now*, choosing
among arms available *now*. Calendar era is therefore not a nuisance to average over —
it is an **eligibility criterion**. That is precisely the target trial emulation move,
and it converts a hard confounding problem into a cohort definition.

> **Reference.** Hernán MA, Robins JM. "Using Big Data to Emulate a Target Trial When
> a Randomized Trial Is Not Available." *American Journal of Epidemiology*
> 2016;183(8):758–764. [DOI](https://academic.oup.com/aje/article-abstract/183/8/758/1739860) ·
> [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4832051/)

---

## 1. Restrict the primary cohort to an availability-stable window (2018+)

**Recommendation.** Define the analysis cohort as line-records with
`line_start_date >= 2018-01-01`. Retain 2008–2017 data only for the secondary use in §9.

**Motivation.** The treatment mix moves violently across the study period, and no `X_`
or `T_` column encodes calendar time — the model is structurally blind to it. Line-1
shares:

| era | ET+ANTI-CDK | CT+ANTI-ANGIO |
|---|---|---|
| 2008–13 | 0.0% | 13.8% |
| 2014–16 | 1.6% | 9.5% |
| 2017–19 | 41.8% | 0.3% |
| 2020–23 | 63.5% | 0.3% |

The effect on measured overlap is direct and large: adding entry year to the propensity
model makes ET+ANTI-CDK near-impossible (e < 0.01) for **36.5%** of the cohort, versus
**0.4%** era-blind; CT+ANTI-ANGIO goes 3.6% → 20.8%. The apparent overlap in the pooled
data is substantially an artefact of pooling eras — the era-blind model happily assigns
a 2010 patient a ~20% probability of receiving a drug that did not yet exist.

2018 is the right boundary because the modern arm set is already in place by then
(CDK4/6i at 48.9% of line 1 in 2018; ANTI-ANGIO already withdrawn). Within 2018+, the
remaining year-on-year drift is *policy* drift, not *availability* drift — and policy
drift is exactly what conditioning handles (§2).

**Cost.** 9,086 patients / 18,517 line-records across lines 1–4, versus 19,865 / 50,800
for the full period — roughly 36% of records retained. The follow-up cost is treated
separately in §4.

> **References.** Palbociclib received European Commission/EMA approval in **November
> 2016**, which dates the availability discontinuity. The pattern is corroborated
> *within the ESME cohort itself*: CDK4/6 inhibitor use rose from **1.3% to 57.0%**
> between time periods in the HR+/HER2− population — closely matching the 0% → 66%
> trajectory measured here.
> — "Recent treatment and survival trends in older versus younger women with
> HR-positive HER2-negative metastatic breast cancer in the real-life multicenter French
> ESME cohort," *European Journal of Cancer*.
> [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0959804925010524)
>
> See also: "Evolution of overall survival and receipt of new therapies by subtype among
> 20,446 metastatic breast cancer patients in the 2008–2017 ESME cohort," *ESMO Open* 2021.
> [Link](https://www.esmoopen.com/article/S2059-7029(21)00072-7/fulltext)

---

## 2. Add calendar time as a covariate *within* the window

**Recommendation.** Include line start year (or months since cohort start) as a feature
in the encoder, applied only inside the restricted window.

**Motivation.** Once all arms are continuously available across the window, calendar
time becomes a genuine covariate rather than a determinant of support, and conditioning
on it is both safe and sufficient. Residual policy drift within 2018+ is real —
ET+ANTI-CDK still climbs 48.9% → 66.2% between 2018 and 2022 — and it correlates with
secular improvements in supportive care, imaging, and the availability of later-line
options.

**Honest qualifier.** The magnitude of the *outcome*-side secular trend in this specific
subtype is modest. ESME's own time-trend analyses found overall survival gains over
calendar period were largely concentrated in HER2-overexpressing disease, with more
limited gains in HR+/HER2− (on the order of a 2-month real-world PFS improvement). So
the confounding contribution of Problem 2 is probably small here — which *strengthens*
the case that the availability problem (§1) is the dominant issue and restriction is the
primary fix, with this step as insurance rather than the main event.

> **Reference.** Gobbini E, Ezzalfani M, Dieras V, Bachelot T, Brain E, et al.
> "Time trends of overall survival among metastatic breast cancer patients in the
> real-life ESME cohort." *European Journal of Cancer* 2018;96:17–24.
> [PubMed](https://pubmed.ncbi.nlm.nih.gov/29660596/) ·
> [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0959804918307330)

---

## 3. Mask the action set per (era, line), not globally

**Recommendation.** Restrict the recommendable arm set to arms with adequate empirical
support in the patient's era **and line**. Apply the mask at recommendation time and to
the counterfactual heads; do not let the model emit a preference over an unsupported arm.

**Motivation.** Positivity splits the arms into two clean groups, and the split is
line-dependent. Even inside the 2018+ window, line-1 arm counts are:

| arm | n (line 1, 2018+) | recommendable? |
|---|---|---|
| ET+ANTI-CDK wo CT | 3,641 | yes |
| ET alone | 1,070 | yes |
| POLYCT alone | 582 | yes |
| MONOCT std alone | 521 | yes |
| OTHER | 204 | yes (but see §6) |
| NO TREATMENT | 71 | no (see §7) |
| ET+TT | 81 | no |
| CT+ANTI-HER2 | 47 | no (see §8) |
| CT+ANTI-ANGIO | 21 | no |
| CT+IT | 17 | no |
| CT+TT | 2 | no |

And the viable set **shifts by line** — at line 4, ET alone collapses to 90 while MONOCT
std alone dominates at 1,359. A single global action space is therefore wrong in both
directions: it permits recommendations that are unsupported at that line, and it applies
a line-1-appropriate arm list to line-4 decisions.

In the full pooled data the unsupported arms are severe: CT+ANTI-HER2, CT+IT, CT+TT and
NO TREATMENT sit at e_a < 0.01 for **60–99%** of the cohort at every line. Predictions
for these arms are extrapolation, not identification.

> **References.** The standard trimming rule — discard units with estimated propensity
> outside [0.1, 0.9] — and the general principle of redefining the estimand to the
> region of common support:
> Crump RK, Hotz VJ, Imbens GW, Mitnik OA. "Dealing with limited overlap in estimation
> of average treatment effects." *Biometrika* 2009;96(1):187–199.
> [DOI](https://academic.oup.com/biomet/article/96/1/187/235329)
>
> On restricting the action space in learned treatment policies specifically:
> Gottesman O, Johansson F, Komorowski M, Faisal A, Sontag D, Doshi-Velez F, Celi LA.
> "Guidelines for reinforcement learning in healthcare." *Nature Medicine*
> 2019;25:16–18. [DOI](https://www.nature.com/articles/s41591-018-0310-5)

---

## 4. Shorten the evaluation horizons and move to RMST

**Recommendation.** Replace `horizon_times = [100, 75, 50, 30]` in `configs/config.toml:17`
with horizons of **24 and 36 months**, and report restricted mean survival time (RMST)
rather than survival probability at a distant fixed point.

**Motivation.** This is the real price of the era restriction, and it is unavoidable.
The database lock is February 2024, so a cohort starting in 2018 has a *maximum possible*
follow-up of ~74 months. Measured potential (administrative) follow-up for line-1
patients:

| window | n (line 1) | median potential FU | ≥24 mo | ≥36 mo | ≥60 mo |
|---|---|---|---|---|---|
| 2008+ | 19,866 | 105.8 mo | 96.7% | 90.8% | 76.8% |
| 2016+ | 9,012 | 59.3 mo | 92.7% | 79.6% | 48.9% |
| **2018+** | **6,257** | **48.0 mo** | **89.5%** | **70.7%** | **26.4%** |
| 2019+ | 4,804 | 41.1 mo | 86.3% | 61.8% | 4.2% |

In the 2018+ window only 26.4% of patients have 60 months of potential follow-up and
essentially none have 100. The configured 100-month horizon is not estimable there —
any number reported at it would be extrapolation from the tail of the survival curve.

RMST is additionally the better target on its own merits: it is interpretable without
the proportional-hazards assumption (which the era-varying treatment mix makes
particularly untrustworthy), it is a natural quantity to compare across arms for a
recommender, and a line-4 treatment decision does not turn on 100-month survival.

> **References.** Royston P, Parmar MKB. "Restricted mean survival time: an alternative
> to the hazard ratio for the design and analysis of randomized trials with a
> time-to-event outcome." *BMC Medical Research Methodology* 2013;13:152.
> [DOI](https://link.springer.com/article/10.1186/1471-2288-13-152)
>
> For quantifying follow-up correctly (use reverse Kaplan–Meier rather than median
> observed time, which is downward-biased):
> Schemper M, Smith TL. "A note on quantifying follow-up in studies of failure time."
> *Controlled Clinical Trials* 1996;17:343–346.
> [PubMed](https://pubmed.ncbi.nlm.nih.gov/8889347/)

---

## 5. Validate on a temporal split, not a random one

**Recommendation.** Train on 2018–2021, test on 2022–2023. Report this as the primary
validation; keep random K-fold CV only as a secondary, clearly-labelled optimistic bound.

**Motivation.** Random cross-validation across calendar time leaks the treatment-policy
era: a 2019 record in the training fold tells the model what the 2019 policy looked like,
inflating apparent performance on 2019 test records in a way that will not reproduce at
deployment. Since the treatment mix is the thing that drifts most, this is not a
hypothetical leak.

More importantly, the estimand defined at `first_draft.tex:155` explicitly rests on
"the deployment-time future-treatment policy remaining close to the one observed in the
training data." Era drift *is* the failure mode of that assumption, and a temporal split
is the only validation design that tests it. Reporting it turns a stated caveat into a
measured quantity.

> **References.** Finlayson SG, Subbaswamy A, Singh K, Bowers J, Kupke A, Zittrain J,
> Kohane IS, Saria S. "The Clinician and Dataset Shift in Artificial Intelligence."
> *New England Journal of Medicine* 2021;385(3):283–286.
> [DOI](https://www.nejm.org/doi/full/10.1056/NEJMc2104626) ·
> [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8665481/)

---

## 6. Qualify the consistency assumption — and consider re-grouping `OTHER`

**Recommendation.** Soften the claim at `first_draft.tex:169` that treatment lines
"correspond to well-defined therapeutic regimens with no ambiguity." Either split or
exclude the heterogeneous arms, and state the versions-of-treatment caveat explicitly
for those retained.

**Motivation.** The data contradicts the current claim for several arms. Counting
distinct drug-flag combinations within each category (lines 1–4):

| arm | n | distinct regimens | modal share | regimens for 80% of mass |
|---|---|---|---|---|
| OTHER | 7,214 | **736** | 0.16 | **67** |
| ET+TT | 1,705 | **344** | 0.16 | **76** |
| POLYCT alone | 7,652 | 354 | 0.19 | 22 |
| ET+ANTI-CDK wo CT | 7,368 | 176 | 0.26 | 11 |
| MONOCT std alone | 11,356 | 132 | 0.35 | 8 |
| ET alone | 12,495 | 146 | 0.33 | 8 |
| CT+ANTI-ANGIO | 2,352 | 101 | 0.65 | 4 |

`OTHER` requires 67 distinct regimens to cover 80% of its mass, with a modal regimen
accounting for only 16%. "Set A_k = OTHER" is not a well-defined intervention, and the
estimated counterfactual under it is an average over a version distribution that will
not be stable at deployment. `CT+ANTI-ANGIO`, by contrast, is genuinely well-defined
(modal share 0.65, four regimens for 80% of mass) — the problem is arm-specific, not
cohort-wide, and the write-up should say so rather than making a blanket claim in either
direction.

The formal machinery for this case exists: effects remain identifiable under multiple
versions provided the estimand is understood as an effect of the *version distribution*,
which requires that distribution to be stable between study and deployment.

> **References.** VanderWeele TJ, Hernán MA. "Causal inference under multiple versions
> of treatment." *Journal of Causal Inference* 2013;1(1):1–20. DOI: 10.1515/jci-2012-0002.
> [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4219328/) ·
> [Publisher](https://www.degruyterbrill.com/document/doi/10.1515/jci-2012-0002/html)

---

## 7. Drop `NO TREATMENT` as a treatment arm

**Recommendation.** Exclude `NO TREATMENT` from the action space; treat those records as
a cohort-definition matter rather than an arm.

**Motivation.** Its counts across lines 1–4 are **207 / 0 / 0 / 1**. It is not a
treatment option at lines 2+ — it appears to be a line-1 coding artefact (patients
recorded at metastatic diagnosis before any therapy started). Modelling it as an arm
means the counterfactual heads produce a survival prediction for "assign no treatment"
at line 3, a quantity with essentially zero empirical support (e_a < 0.01 for 61% of the
cohort even at line 1). This is structural non-positivity of the cleanest kind.

> **Reference.** Petersen et al. 2012 (as above) — "modification of the target
> intervention" is the prescribed response when an intervention level has no support.

---

## 8. Resolve the `CT+ANTI-HER2` records in an HR+HER2− cohort

**Recommendation.** Data question to settle before the next training run: 197
`CT+ANTI-HER2` line-records (lines 1–4) sit inside a cohort defined as HER2-negative.

**Motivation.** This is either (a) legitimate — HER2-low patients receiving
antibody–drug conjugates, which would be clinically coherent in later lines and recent
years — or (b) a subtype-assignment leak that also calls the cohort definition into
question. `X_her2status` is 4-valued (1/2/3/5) and its coding was not resolved during
this audit; the 197 records distribute across levels 1/2/3 rather than concentrating in
one, which does not obviously support either explanation. Worth a direct check with
whoever built the subtype flags. Either way the arm fails the positivity filter in §3,
so this does not block the modelling work — but it may indicate a labelling problem with
wider consequences.

---

## 9. Optional: pretrain the encoder on the full period, fit heads on the window

**Recommendation.** Secondary/exploratory only. Pretrain the shared encoder on all
2008+ data, then fit the treatment and outcome heads on the 2018+ window.

**Motivation.** Prognostic structure (how metastatic burden, histology, and progression
history map to risk) is plausibly era-stable, whereas treatment effects are not. This
would recover statistical efficiency from the 64% of records that §1 discards.

**Why secondary.** The era-stability of prognostic structure is an assumption, not an
established fact, and this design adds a leakage path — outcome information from the
excluded era can reach the heads through the representation. 9,086 patients is a
workable primary cohort; do not accept this complexity until the simple version is shown
to be underpowered.

---

## 10. Validation opportunity: the CDK4/6i introduction as a natural experiment

**Recommendation.** Use the abrupt 2016–2018 CDK4/6 inhibitor introduction as an
external check on the model's estimated CDK effect.

**Motivation.** The discontinuity is sharp, exogenous to individual patient
characteristics (driven by regulatory approval and reimbursement, not by clinical
state), and large in magnitude — close to an ideal calendar-time instrument. An effect
estimate derived from the pre/post discontinuity does **not** rely on sequential
ignorability, so agreement with the model's estimate is meaningful corroboration of the
part of the analysis that is otherwise untestable. This is cheap to run relative to its
evidential value, and would materially strengthen the paper.

---

## What none of this fixes

Everything above addresses **positivity** (§1, §3, §7), **consistency** (§6), and the
calendar-time component of confounding (§2). **Sequential ignorability is untouched.**
Restricting the window removes the availability problem and the secular trend; it does
nothing about unmeasured confounders operating *within* the window — physician gestalt,
performance status if unrecorded, patient preference, frailty.

This matters because the project's own semi-synthetic benchmark
(`scripts/semisynthetic/`) established precisely this: observed confounding does not
bias g-formula-style estimators when the confounders are recorded and the hazard form is
approximately right, whereas **hidden** confounding produces monotone bias. Unmeasured
confounding is the failure mode that actually breaks these methods.

**Recommendation.** Report a quantitative sensitivity analysis for unmeasured
confounding alongside the main results, so the untestable assumption is at least
bounded rather than merely asserted.

> **Reference.** VanderWeele TJ, Ding P. "Sensitivity Analysis in Observational
> Research: Introducing the E-Value." *Annals of Internal Medicine* 2017;167(4):268–274.
> [DOI](https://www.acpjournals.org/doi/abs/10.7326/M16-2607)

---

## Implementation checklist

| # | Change | Where |
|---|---|---|
| 1 | Cohort filter `line_start_date >= 2018` | `src/CausalSurv/data/datamodule_cv.py`, new config key |
| 2 | Calendar-time feature | dataset feature construction |
| 3 | Per-(era, line) action mask | model heads + recommendation logic |
| 4 | `horizon_times = [36, 24]`, RMST metric | `configs/config.toml:17`, `evaluation/evaluator.py` |
| 5 | Temporal train/test split | `datamodule_cv.py` split logic |
| 6 | Re-group or exclude `OTHER`, `ET+TT` | preprocessing; `first_draft.tex:169` |
| 7 | Exclude `NO TREATMENT` arm | preprocessing |
| 8 | Resolve HER2 labelling question | data provenance — external |

---

## Caveat on the diagnostics

Propensities were estimated with multinomial logistic regression, which is linear in the
features. A more flexible model could only separate the arms more sharply, so the
overlap violations reported here are a **lower bound** on severity — the true positivity
problem is at least this bad, possibly worse.
