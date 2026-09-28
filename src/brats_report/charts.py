"""Draw the README charts from the evaluation JSONs (needs matplotlib only).

    python -m brats_report.charts --eval docs/eval/01_baseline.json docs/eval/02_tta.json ... \\
        --classifier docs/eval/classifier.json --out docs/images

``--eval`` files are drawn in the order given (one step per file).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REGIONS = ("WT", "TC", "ET")
REGION_LABELS = {"WT": "Whole tumour", "TC": "Tumour core", "ET": "Enhancing"}

# Colours: categorical slots in fixed order, a one-hue sequential ramp, text inks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQ = ("#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _name(e: dict) -> str:
    """Step name without the post-processing suffix, e.g. "80 patients [largest]"."""
    return e["name"].split(" [")[0]


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _figure(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _title(fig, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=13, fontweight="bold", color=INK,
             va="top", ha="left")
    fig.text(0.012, 0.915, subtitle, fontsize=9.5, color=INK_2, va="top", ha="left")


def dice_steps(evals: list[dict], out: Path) -> Path:
    """Grouped columns: mean 3-D Dice per region for each step, with the
    interquartile range across patients as a whisker (means hide the spread)."""
    fig, ax = _figure(9, 4.8)
    n, width = len(evals), 0.24
    x = np.arange(n)
    for i, r in enumerate(REGIONS):
        vals = [e["summary"][r]["dice_mean"] for e in evals]
        per_case = [[c["scores"][r]["dice"] for c in e["cases"]] for e in evals]
        q25, q75 = (np.array([np.percentile(d, q) for d in per_case]) for q in (25, 75))
        xs = x + (i - 1) * width
        ax.bar(xs, vals, width * 0.9, color=SERIES[i], label=REGION_LABELS[r], zorder=3)
        ax.vlines(xs, q25, q75, color=INK_2, linewidth=1, zorder=4)
        ax.hlines(np.r_[q25, q75], np.r_[xs, xs] - 0.03, np.r_[xs, xs] + 0.03,
                  color=INK_2, linewidth=1, zorder=4)
        for xi, v, top in zip(xs, vals, q75, strict=True):
            ax.text(xi, max(v, top) + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [_name(e) for e in evals], fontsize=9, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Dice per patient (whole scan)", color=INK_2, fontsize=9)
    _style(ax)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.005, 0.86), fontsize=9, labelcolor=INK)
    n_cases = evals[-1]["n_cases"]
    _title(fig, "Segmentation quality, step by step",
           f"The same {n_cases} held-out patients (never used for training); bars are means, "
           "whiskers the middle half of patients")
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.07, right=0.99)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def volume_agreement(first: dict, last: dict, out: Path, region="WT") -> Path:
    """Scatter: predicted vs expert tumour volume, first step vs final step."""
    fig, ax = _figure(6.2, 5.6)
    gmax = 0.0
    for e, color, marker in ((first, SERIES[1], "o"), (last, SERIES[0], "o")):
        gv = np.array([c["scores"][region]["gt"]["volume_cm3"] for c in e["cases"]])
        pv = np.array([c["scores"][region]["pred"]["volume_cm3"] for c in e["cases"]])
        err = e["summary"][region]["volume_median_abs_pct"]
        ax.scatter(gv, pv, s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   marker=marker, label=f"{_name(e)}  (median error {err}%)")
        gmax = max(gmax, gv.max(), pv.max())
    lim = gmax * 1.08
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, zorder=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Expert mask volume (cm³)", color=INK_2, fontsize=9)
    ax.set_ylabel("Volume in the AI report (cm³)", color=INK_2, fontsize=9)
    _style(ax, grid_axis="both")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=INK)
    _title(fig, f"{REGION_LABELS[region]} volume: report vs. expert",
           f"One dot per held-out patient ({last['n_cases']}). On the diagonal the report "
           "matches the expert;" + chr(10) + "below it, the tumour was under-measured.")
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.12, right=0.97)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def confusion(metrics: dict, out: Path) -> Path:
    """Confusion matrix of the tumour-type classifier (row-normalised colour, counts)."""
    cm = np.array(metrics["confusion_matrix"])
    classes = metrics["classes"]
    names = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour",
             "pituitary": "Pituitary"}
    rate = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = _figure(6.0, 5.4)
    k = len(classes)
    for i in range(k):
        for j in range(k):
            step = int(round(rate[i, j] * (len(SEQ) - 1)))
            step = max(step, 1) if cm[i, j] else 0
            color = SEQ[step]
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, color=color, zorder=2))
            txt = INK if _luminance(color) > 0.3 else "#ffffff"
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}", ha="center", va="center", fontsize=11,
                    color=txt, fontweight="bold" if i == j else "normal", zorder=3)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    labels = [names.get(c, c) for c in classes]
    ax.set_xticks(np.arange(k) + 0.5, labels, fontsize=9, color=INK)
    ax.set_yticks(np.arange(k) + 0.5, labels, fontsize=9, color=INK)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("True", color=INK_2, fontsize=9)
    _style(ax, grid_axis=None)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    t = metrics["test"]
    _title(fig, "Tumour type: held-out test set",
           f"{metrics['n_test']} images · accuracy {t['accuracy']:.1%} · "
           f"macro F1 {t['macro_f1']:.3f}")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.2, right=0.97)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def markdown_table(evals: list[dict], final: int = -1) -> str:
    """The README's results table, one row per step (bold = the ``final`` step)."""
    head = ("| step | Dice WT / TC / ET | WT HD95 | WT volume error (median) | "
            "WT volume bias | WT max 3-D diameter error | WT longest axial error |")
    rows = [head, "|" + "---|" * 7]
    for i, e in enumerate(evals):
        s, w = e["summary"], e["summary"]["WT"]
        cells = [_name(e),
                 " / ".join(f"{s[r]['dice_mean']:.2f}" for r in REGIONS),
                 f"{w['hd95_median_mm']:.1f} mm",
                 f"{w['volume_median_abs_pct']:.1f} %",
                 f"{w['volume_bias_cm3']:+.1f} cm³",
                 f"{w['max_diameter_mae_mm']:.1f} mm",
                 f"{w['recist_mae_mm']:.1f} mm"]
        if i == final % len(evals):
            cells = [f"**{c}**" for c in cells]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="*", default=[], help="evaluate.py JSONs, one per step")
    ap.add_argument("--classifier", default="", help="classifier metrics JSON")
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--final", type=int, default=0,
                    help="1-based step shipped in the app (bold in the table, compared with "
                         "step 1 in the volume chart); default: the last step")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if args.eval:
        evals = [json.loads(Path(p).read_text()) for p in args.eval]
        written.append(dice_steps(evals, out / "dice_steps.png"))
        final = args.final - 1 if args.final else len(evals) - 1
        written.append(volume_agreement(evals[0], evals[final], out / "volume_agreement.png"))
        print(markdown_table(evals, final) + "\n")
    if args.classifier:
        written.append(confusion(json.loads(Path(args.classifier).read_text()),
                                 out / "classifier_confusion.png"))
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
