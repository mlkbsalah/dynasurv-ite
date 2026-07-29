"""Repetition of treatment category over the first 4 treatment lines.

For the HR+HER2- cohort, restricts to patients with at least 4 treatment lines,
takes lines 1-4, and counts every time a patient is given a
``T_treatment_category`` they have already had. Each such repetition is one of
two kinds:

  * **consecutive** - the same category in the immediately next line
    (a continuation, e.g. A A);
  * **non-consecutive** - the category returns after the patient switched away
    (a rechallenge, e.g. A B A).

Two interactive figures, each three panels, sharing one colour language:

  * ``treatment_category_repeats.html`` - both kinds side by side: how patients
    repeat, which categories they repeat, and the full set of 15 sequence
    patterns four lines can take.
  * ``treatment_category_repeats_consecutive.html`` - consecutive repeats only,
    in more depth: blocks per patient, category by run length, and where in the
    sequence the blocks sit.

Hover any bar/segment for the underlying counts and percentages.

Run:  python data_analysis/treatment_category_repeats.py
Figures -> data_analysis/plots/
"""

from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
N_LINES = 4

# ---- palette / chrome (dataviz reference palette, light mode) ----
SURFACE, INK, SEC, MUTED, GRID, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# One colour language for the *kind* of repetition, used by every panel of the
# combined figure. "none" is deliberately the neutral ink token, not a hue.
KINDS = ("none", "consec", "nonconsec", "both")
KIND_LABEL = {
    "none": "no repetition",
    "consec": "consecutive",
    "nonconsec": "non-consecutive",
    "both": "both kinds",
}
KIND_COLOR = {
    "none": MUTED,
    "consec": "#2a78d6",  # categorical slot 1
    "nonconsec": "#eb6834",  # categorical slot 2
    "both": "#1baf7a",  # categorical slot 3
}
# Run length is ordinal, so the consecutive-only figure encodes it as one hue
# light -> dark rather than as separate categorical hues.
LENGTHS = (2, 3, 4)
LEN_COLOR = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}
LEN_LABEL = {n: f"{n} lines" for n in LENGTHS}
N_LABELLED = 5  # direct-label only the largest few sequence patterns


class Repeats(NamedTuple):
    """Everything both figures are drawn from."""

    n_pat: int
    profile: Counter  # repetition kind -> n patients
    patterns: Counter  # canonical sequence pattern -> n patients
    by_cat: dict  # category -> {consec, nonconsec} -> n repetitions
    runlen: dict  # category -> run length -> n consecutive blocks
    gaplen: dict  # category -> lines skipped -> n rechallenges
    blocks: Counter  # n consecutive blocks -> n patients
    shapes: Counter  # (first line, run length) -> n consecutive blocks


def canonical(seq):
    """Relabel a category sequence as A/B/C/... in order of first appearance."""
    letters = {}
    return "".join(letters.setdefault(c, chr(65 + len(letters))) for c in seq)


def positions(seq):
    """Map each distinct element to its list of 0-based positions in ``seq``."""
    pos = defaultdict(list)
    for i, c in enumerate(seq):
        pos[c].append(i)
    return pos


def kind_of(seq):
    """Classify a sequence by which kinds of repetition it contains."""
    gaps = [b - a for ps in positions(seq).values() for a, b in zip(ps, ps[1:])]
    consec, nonconsec = any(g == 1 for g in gaps), any(g > 1 for g in gaps)
    if consec and nonconsec:
        return "both"
    return "consec" if consec else "nonconsec" if nonconsec else "none"


def pattern_gloss(pat):
    """Plain-English description of which lines share a category."""
    shared = [ps for ps in positions(pat).values() if len(ps) > 1]
    if not shared:
        return "four different categories"
    return " · ".join("lines " + "+".join(str(i + 1) for i in ps) for ps in shared)


def runs(seq):
    """Yield ``(category, first 0-based position, length)`` for each run in ``seq``."""
    start = 0
    for cat, run in groupby(seq):
        length = sum(1 for _ in run)
        yield cat, start, length
        start += length


def load_repeats():
    """Read the cohort and tally every repetition, both kinds."""
    df = pd.read_parquet(DATA_PATH)

    # drop patients with duplicated line numbers (data anomaly)
    bad = df.groupby("usubjid")["X_line_number"].apply(lambda s: s.duplicated().any())
    df = df[~df["usubjid"].isin(bad[bad].index)].copy()

    # keep patients reaching >= N_LINES, restrict to the first N_LINES lines
    maxline = df.groupby("usubjid")["X_line_number"].max()
    keep = maxline[maxline >= N_LINES].index
    sub = df[df["usubjid"].isin(keep) & (df["X_line_number"] <= N_LINES)]
    sub = sub.sort_values(["usubjid", "X_line_number"])

    profile, patterns, blocks, shapes = Counter(), Counter(), Counter(), Counter()
    by_cat, runlen, gaplen = (defaultdict(Counter) for _ in range(3))
    for _, grp in sub.groupby("usubjid"):
        seq = list(grp["T_treatment_category"])
        profile[kind_of(seq)] += 1
        patterns[canonical(seq)] += 1
        for cat, ps in positions(seq).items():
            for a, b in zip(ps, ps[1:]):
                by_cat[cat]["consec" if b - a == 1 else "nonconsec"] += 1
                if b - a > 1:
                    gaplen[cat][b - a - 1] += 1
        n_blocks = 0
        for cat, start, length in runs(seq):
            if length >= 2:
                runlen[cat][length] += 1
                shapes[(start + 1, length)] += 1
                n_blocks += 1
        blocks[n_blocks] += 1

    return Repeats(
        sub["usubjid"].nunique(),
        profile,
        patterns,
        by_cat,
        runlen,
        gaplen,
        blocks,
        shapes,
    )


# --------------------------------------------------------------------------- #
# shared chrome
# --------------------------------------------------------------------------- #
def _detail(counter, unit):
    """Render a small distribution as one hover line, e.g. ``12 ×2 lines, 3 ×3``."""
    if not counter:
        return "—"
    return ", ".join(f"{n:,} {unit(k)}" for k, n in sorted(counter.items()))


def _new_figure(titles):
    """Two rows: two panels on top, one full-width panel underneath."""
    return make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        column_widths=[0.34, 0.66],
        row_heights=[0.44, 0.56],
        horizontal_spacing=0.13,
        vertical_spacing=0.17,
        subplot_titles=titles,
    )


def _apply_chrome(fig, title, subtitle, legend_title):
    """Palette, typography, title block and a legend parked inside panel C."""
    fig.update_layout(
        barmode="stack",
        bargap=0.35,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif",
            color=INK,
            size=12,
        ),
        title=dict(
            text=f"{title}<br><span style='font-size:13px;color:{SEC}'>{subtitle}</span>",
            x=0.012,
            xanchor="left",
            font=dict(size=19),
        ),
        # panel C's upper-right is empty in both figures, and it is the panel
        # that uses every colour, so the legend belongs there
        legend=dict(
            title_text=legend_title,
            title_font_color=SEC,
            x=0.995,
            xanchor="right",
            y=0.45,
            yanchor="top",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        margin=dict(l=10, r=20, t=164, b=60),
        height=880,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=BASE,
            font=dict(color=INK, size=12),
        ),
    )
    # left-align each subplot title to its own panel
    for i, ann in enumerate(fig.layout.annotations[:3]):
        axis = fig.layout["xaxis" if i == 0 else f"xaxis{i + 1}"]
        ann.update(x=axis.domain[0], xanchor="left")


def _end_labels(fig, cats, totals):
    """Row totals at the end of each stacked horizontal bar in panel B."""
    xmax = max(totals.values())
    for c in cats:
        fig.add_annotation(
            x=totals[c] + xmax * 0.01,
            y=c,
            text=f"{totals[c]:,}",
            xref="x2",
            yref="y2",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(color=INK, size=11),
        )
    return xmax


def _write(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.write_html(
        out,
        include_plotlyjs=True,  # self-contained (works offline / in sandboxes)
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return out


# --------------------------------------------------------------------------- #
# figure 1 - both kinds of repetition
# --------------------------------------------------------------------------- #
def add_profile_panel(fig, n_pat, profile):
    """Panel A: every patient in exactly one repetition class."""
    for kind in KINDS:
        n = profile.get(kind, 0)
        fig.add_trace(
            go.Bar(
                x=[KIND_LABEL[kind]],
                y=[100 * n / n_pat],
                name=KIND_LABEL[kind],
                legendgroup=kind,
                marker_color=KIND_COLOR[kind],
                width=0.62,
                text=[f"{100 * n / n_pat:.1f}%"],
                textposition="outside",
                textfont=dict(color=INK, size=12),
                customdata=[[n]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{customdata[0]:,} patients<br>"
                    "%{y:.1f}% of patients"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def add_category_panel(fig, by_cat, runlen, gaplen):
    """Panel B: repetitions per category, split by kind."""
    totals = {c: sum(v.values()) for c, v in by_cat.items()}
    cats = sorted(totals, key=totals.get)  # ascending -> largest on top
    detail = {
        "consec": [
            "run lengths: " + _detail(runlen[c], lambda k: f"×{k} lines") for c in cats
        ],
        "nonconsec": [
            "gaps: " + _detail(gaplen[c], lambda k: f"after {k} line(s) away")
            for c in cats
        ],
    }
    for kind in ("consec", "nonconsec"):
        vals = [by_cat[c][kind] for c in cats]
        fig.add_trace(
            go.Bar(
                y=cats,
                x=vals,
                orientation="h",
                name=KIND_LABEL[kind],
                legendgroup=kind,
                showlegend=False,
                marker_color=KIND_COLOR[kind],
                marker_line=dict(color=SURFACE, width=1.2),
                customdata=[
                    [100 * v / totals[c], totals[c], d]
                    for v, c, d in zip(vals, cats, detail[kind])
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "%{fullData.name}: %{x:,} repetitions "
                    "(%{customdata[0]:.1f}% of this category)<br>"
                    "%{customdata[2]}<br>"
                    "category total: %{customdata[1]:,} repetitions"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    return cats, totals


def add_pattern_panel(fig, n_pat, patterns):
    """Panel C: all 15 sequence patterns four lines can take.

    One trace per kind rather than one trace with per-point colours, so that
    clicking a legend entry filters this panel along with the other two.
    """
    items = patterns.most_common()
    labelled = {p for p, _ in items[:N_LABELLED]}
    for kind in KINDS:
        sel = [(p, n) for p, n in items if kind_of(p) == kind]
        if not sel:
            continue
        fig.add_trace(
            go.Bar(
                x=[" ".join(p) for p, _ in sel],
                y=[100 * n / n_pat for _, n in sel],
                name=KIND_LABEL[kind],
                legendgroup=kind,
                showlegend=False,
                marker_color=KIND_COLOR[kind],
                width=0.66,
                text=[
                    f"{100 * n / n_pat:.1f}%" if p in labelled else "" for p, n in sel
                ],
                textposition="outside",
                textfont=dict(color=INK, size=11),
                customdata=[[n, pattern_gloss(p)] for p, n in sel],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "%{customdata[0]:,} patients (%{y:.1f}%)<br>"
                    "%{fullData.name} — %{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    order = [" ".join(p) for p, _ in items]
    return order, 100 * items[0][1] / n_pat


def make_figure(rep):
    """Build the combined (both kinds) figure and write it to an HTML file."""
    fig = _new_figure(
        (
            "<b>How patients repeat a category</b>",
            "<b>Which categories are repeated — and how</b>",
            "<b>Every sequence four lines can take</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   A/B/C/D = distinct categories, in order of first appearance</span>",
        )
    )

    add_profile_panel(fig, rep.n_pat, rep.profile)
    cats, totals = add_category_panel(fig, rep.by_cat, rep.runlen, rep.gaplen)
    pat_order, pct_max = add_pattern_panel(fig, rep.n_pat, rep.patterns)
    xmax = _end_labels(fig, cats, totals)

    n_consec = sum(v["consec"] for v in rep.by_cat.values())
    n_non = sum(v["nonconsec"] for v in rep.by_cat.values())
    pct_any = 100 * (1 - rep.profile.get("none", 0) / rep.n_pat)
    _apply_chrome(
        fig,
        "Repetition of treatment category over the first 4 lines",
        (
            f"HR+HER2− cohort · patients with ≥ 4 treatment lines (n = {rep.n_pat:,}) · "
            "a repetition = a category given again, either in the next line "
            "(<b>consecutive</b>) or after switching away (<b>non-consecutive</b>)"
            "<br>"
            f"{pct_any:.0f}% repeat ≥ 1 category · "
            f"{n_consec + n_non:,} repetitions total "
            f"({n_consec:,} consecutive, {n_non:,} non-consecutive)"
        ),
        "kind of repetition",
    )

    # Panel A
    fig.update_yaxes(
        title_text="% of patients",
        gridcolor=GRID,
        zeroline=False,
        range=[0, 100 * max(rep.profile.values()) / rep.n_pat * 1.16],
        row=1,
        col=1,
    )
    fig.update_xaxes(type="category", gridcolor=GRID, row=1, col=1)
    # Panel B
    fig.update_xaxes(
        title_text="number of repetitions",
        gridcolor=GRID,
        zeroline=False,
        range=[0, xmax * 1.12],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=cats,
        gridcolor="rgba(0,0,0,0)",
        row=1,
        col=2,
    )
    # Panel C
    fig.update_yaxes(
        title_text="% of patients",
        gridcolor=GRID,
        zeroline=False,
        range=[0, pct_max * 1.14],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="category sequence over lines 1 → 4",
        type="category",
        categoryorder="array",
        categoryarray=pat_order,
        gridcolor=GRID,
        tickfont=dict(family="SFMono-Regular, Menlo, monospace", size=12),
        row=2,
        col=1,
    )
    return _write(fig, "treatment_category_repeats.html")


# --------------------------------------------------------------------------- #
# figure 2 - consecutive repeats only
# --------------------------------------------------------------------------- #
def add_block_count_panel(fig, n_pat, blocks):
    """Panel A: how many consecutive blocks a patient has (0, 1 or 2)."""
    for k in sorted(blocks):
        n = blocks[k]
        fig.add_trace(
            go.Bar(
                x=[str(k)],
                y=[100 * n / n_pat],
                marker_color=MUTED if k == 0 else KIND_COLOR["consec"],
                width=0.5,
                text=[f"{100 * n / n_pat:.1f}%"],
                textposition="outside",
                textfont=dict(color=INK, size=12),
                customdata=[[n]],
                hovertemplate=(
                    "<b>%{x} consecutive block(s)</b><br>"
                    "%{customdata[0]:,} patients<br>"
                    "%{y:.1f}% of patients"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )


def add_runlen_panel(fig, runlen):
    """Panel B: consecutive blocks per category, split by run length."""
    totals = {c: sum(v.values()) for c, v in runlen.items()}
    cats = sorted(totals, key=totals.get)  # ascending -> largest on top
    for length in LENGTHS:
        vals = [runlen[c][length] for c in cats]
        fig.add_trace(
            go.Bar(
                y=cats,
                x=vals,
                orientation="h",
                name=LEN_LABEL[length],
                legendgroup=length,
                marker_color=LEN_COLOR[length],
                marker_line=dict(color=SURFACE, width=1.2),
                customdata=[
                    [100 * v / totals[c], totals[c], v * (length - 1)]
                    for v, c in zip(vals, cats)
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "%{x:,} blocks of %{fullData.name} "
                    "(%{customdata[0]:.1f}% of this category)<br>"
                    "= %{customdata[2]:,} repetitions<br>"
                    "category total: %{customdata[1]:,} blocks"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    return cats, totals


def add_shape_panel(fig, shapes):
    """Panel C: where in the sequence the blocks sit, ordered by run length."""
    total = sum(shapes.values())
    order = sorted(shapes, key=lambda s: (s[1], s[0]))
    labels = {s: f"{s[0]}–{s[0] + s[1] - 1}" for s in order}
    for length in LENGTHS:
        sel = [s for s in order if s[1] == length]
        if not sel:
            continue
        fig.add_trace(
            go.Bar(
                x=[labels[s] for s in sel],
                y=[100 * shapes[s] / total for s in sel],
                name=LEN_LABEL[length],
                legendgroup=length,
                showlegend=False,
                marker_color=LEN_COLOR[length],
                width=0.42,
                text=[f"{100 * shapes[s] / total:.1f}%" for s in sel],
                textposition="outside",
                textfont=dict(color=INK, size=11),
                customdata=[
                    [shapes[s], f"same category in lines {s[0]}–{s[0] + s[1] - 1}"]
                    for s in sel
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[0]:,} blocks (%{y:.1f}% of all blocks)<br>"
                    "run length: %{fullData.name}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
    return [labels[s] for s in order], 100 * max(shapes.values()) / total


def make_consecutive_figure(rep):
    """Build the consecutive-only figure and write it to an HTML file."""
    fig = _new_figure(
        (
            "<b>How many consecutive blocks a patient has</b>",
            "<b>Which categories are continued — and for how long</b>",
            "<b>Where the block sits in the sequence</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   lines covered by one uninterrupted run of the same category</span>",
        )
    )

    add_block_count_panel(fig, rep.n_pat, rep.blocks)
    cats, totals = add_runlen_panel(fig, rep.runlen)
    shape_order, pct_max = add_shape_panel(fig, rep.shapes)
    xmax = _end_labels(fig, cats, totals)

    n_blocks = sum(totals.values())
    n_reps = sum(v["consec"] for v in rep.by_cat.values())
    pct_any = 100 * (1 - rep.blocks.get(0, 0) / rep.n_pat)
    pooled = Counter()
    for v in rep.runlen.values():
        pooled.update(v)
    _apply_chrome(
        fig,
        "Consecutive repetition of treatment category over the first 4 lines",
        (
            f"HR+HER2− cohort · patients with ≥ 4 treatment lines (n = {rep.n_pat:,}) · "
            "a <b>block</b> = one uninterrupted run of the same category in "
            "adjacent lines"
            "<br>"
            f"{pct_any:.0f}% have ≥ 1 block · "
            f"{n_blocks:,} blocks total "
            f"({_detail(pooled, lambda k: f'of {k} lines')}) "
            f"= {n_reps:,} consecutive repetitions"
        ),
        "run length",
    )

    # Panel A
    fig.update_yaxes(
        title_text="% of patients",
        gridcolor=GRID,
        zeroline=False,
        range=[0, 100 * max(rep.blocks.values()) / rep.n_pat * 1.16],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="consecutive blocks per patient",
        type="category",
        gridcolor=GRID,
        row=1,
        col=1,
    )
    # Panel B
    fig.update_xaxes(
        title_text="number of consecutive blocks",
        gridcolor=GRID,
        zeroline=False,
        range=[0, xmax * 1.12],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=cats,
        gridcolor="rgba(0,0,0,0)",
        row=1,
        col=2,
    )
    # Panel C
    fig.update_yaxes(
        title_text="% of consecutive blocks",
        gridcolor=GRID,
        zeroline=False,
        range=[0, pct_max * 1.16],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="lines covered by the block",
        type="category",
        categoryorder="array",
        categoryarray=shape_order,
        gridcolor=GRID,
        row=2,
        col=1,
    )
    return _write(fig, "treatment_category_repeats_consecutive.html")


def main():
    rep = load_repeats()
    n_consec = sum(v["consec"] for v in rep.by_cat.values())
    n_non = sum(v["nonconsec"] for v in rep.by_cat.values())
    print(f"patients with >= {N_LINES} lines: {rep.n_pat:,}")
    print(f"repeat >= 1 category: {100 * (1 - rep.profile['none'] / rep.n_pat):.1f}%")
    for kind in KINDS:
        n = rep.profile.get(kind, 0)
        print(f"  {KIND_LABEL[kind]:<16} {n:>6,}  ({100 * n / rep.n_pat:5.1f}%)")
    print(f"repetitions: {n_consec:,} consecutive, {n_non:,} non-consecutive")

    print("\ncategory                consecutive  non-consecutive")
    for c in sorted(rep.by_cat, key=lambda c: -sum(rep.by_cat[c].values())):
        print(
            f"  {c:<22} {rep.by_cat[c]['consec']:>10,} {rep.by_cat[c]['nonconsec']:>16,}"
        )

    print("\npattern   patients    kind")
    for pat, n in rep.patterns.most_common():
        print(f"  {' '.join(pat)}  {n:>7,}    {KIND_LABEL[kind_of(pat)]}")

    n_blocks = sum(rep.shapes.values())
    print(f"\nconsecutive blocks: {n_blocks:,}")
    print(f"blocks per patient: {dict(sorted(rep.blocks.items()))}")
    print("  lines   blocks   % of blocks")
    for shape in sorted(rep.shapes, key=lambda s: (s[1], s[0])):
        n = rep.shapes[shape]
        span = f"{shape[0]}-{shape[0] + shape[1] - 1}"
        print(f"  {span:<7} {n:>6,}   {100 * n / n_blocks:>6.1f}%")

    for out in (make_figure(rep), make_consecutive_figure(rep)):
        print("saved", out)


if __name__ == "__main__":
    main()
