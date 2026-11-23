"""Publication figure fig_eval_results.pdf (IEEE double-column width) from the test-split results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths  # noqa: E402

PAGE_W: float = 7.16
TEST_DIR: Path = paths.RESULTS_DIR / "test"
BAR_ARCHS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A3x", "A4")
LABELS: dict[str, str] = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3", "A3x": "A3x", "A4": "A4"}
HONEST_C, LIE_C = "#1565c0", "#c62828"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    g = summary[(summary["scenario"] == "ALL") & summary["architecture"].isin(BAR_ARCHS)].set_index("architecture").loc[list(BAR_ARCHS)]
    x = np.arange(len(g))
    w = 0.38
    for off, metric, color, name in ((-w / 2, "recall", LIE_C, "Recall"), (w / 2, "fpr", HONEST_C, "FPR")):
        val = 100 * g[metric].to_numpy()
        lo, hi = 100 * g[f"{metric}_ci_lo"].to_numpy(), 100 * g[f"{metric}_ci_hi"].to_numpy()
        ax.bar(x + off, val, w, color=color, label=name, yerr=np.vstack([val - lo, hi - val]),
               error_kw={"elinewidth": 0.6, "capsize": 1.5})
    ax.set_xticks(x, [LABELS[a] for a in g.index])
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.set_title("(a) Recall and FPR, 95% CI")
    ax.legend(frameon=False, loc="upper left", ncol=2)


def panel_delta(ax: plt.Axes, per_case: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    groups = [("A4-T3", "HONEST"), ("A4-T3", "COMPLIANT_LIE"), ("A3", "HONEST"), ("A3", "COMPLIANT_LIE")]
    for i, (arch, label) in enumerate(groups):
        d = per_case[(per_case["architecture"] == arch) & (per_case["label"] == label)]["delta"].dropna().to_numpy(dtype=float)
        color = HONEST_C if label == "HONEST" else LIE_C
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d + rng.uniform(-0.015, 0.015, len(d)), s=3, alpha=0.45, color=color, linewidths=0)
        if len(d):
            ax.hlines(d.mean(), i - 0.3, i + 0.3, color="black", linewidth=1)
    ax.set_xticks(range(4), ["A4\nhonest", "A4\nlie", "A3\nhonest", "A3\nlie"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"$\Delta(\mathbf{p}_{obs}, \hat{\mathbf{p}})$")
    ax.set_title(r"(b) Premise divergence (Tier 3; bar = mean)")


def panel_slider(ax: plt.Axes, sl: dict[str, Any]) -> None:
    taus = [c["tau"] for c in sl["curves"]["A4"]]
    ax.plot(taus, [c["c_total"] for c in sl["curves"]["A4"]], color=LIE_C, linewidth=1.1, marker="o", markersize=2, label="A4")
    for arch, ls in (("A0", ":"), ("A1", "--"), ("A2", "-.")):
        if arch in sl["flat_baselines"]:
            ax.axhline(sl["flat_baselines"][arch], color="#616161", linestyle=ls, linewidth=0.9, label=arch)
    ax.set_xlabel(r"Threshold $\tau$")
    ax.set_ylabel(r"$C_{total}$ [EUR / decision]")
    p = sl["params"]
    ax.set_title(rf"(c) Financial Slider ($C_{{human}}$={p['c_human_eur']:.0f}, $L_{{biz}}$={p['l_biz_eur']:.0f})")
    ax.legend(frameon=False, ncol=2)


def main() -> None:
    summary = pd.DataFrame(json.loads((TEST_DIR / "summary.json").read_text(encoding="utf-8"))["summary"])
    per_case = pd.read_csv(TEST_DIR / "per_case.csv")
    sl = json.loads((TEST_DIR / "slider.json").read_text(encoding="utf-8"))
    _style()
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    panel_bars(axes[0], summary)
    panel_delta(axes[1], per_case)
    panel_slider(axes[2], sl)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(paths.FIGURES_DIR / "fig_eval_results.pdf")
    fig.savefig(paths.FIGURES_DIR / "fig_eval_results.png", dpi=200)
    plt.close(fig)
    print(f"Figure written to {paths.FIGURES_DIR / 'fig_eval_results.pdf'}")


if __name__ == "__main__":
    main()
