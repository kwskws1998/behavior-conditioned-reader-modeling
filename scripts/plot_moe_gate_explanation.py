#!/usr/bin/env python
"""Create an explanatory figure for the MoE gate analysis."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


CSV_NAMES = {
    "gates": "all_moe_gates.csv",
    "correlations": "all_gate_behavior_correlations.csv",
    "stability": "all_gate_stability.csv",
    "pca": "all_gate_pca.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-analysis", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/moe_gate_explanation.png"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_tables(args.gate_analysis)
    seed = args.seed if args.seed is not None else choose_seed(tables["gates"])
    fig = build_figure(tables, seed=seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=220, bbox_inches="tight")
    pdf_path = args.output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {args.output_path}")
    print(f"Wrote {pdf_path}")


def load_tables(path: Path) -> dict[str, pd.DataFrame]:
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            loaded = {}
            for key, filename in CSV_NAMES.items():
                match = next((name for name in names if name.endswith(filename)), None)
                if match is None:
                    raise FileNotFoundError(f"Missing {filename} in {path}")
                with archive.open(match) as handle:
                    loaded[key] = pd.read_csv(handle)
            return loaded
    if path.is_dir():
        return {key: pd.read_csv(path / filename) for key, filename in CSV_NAMES.items()}
    raise ValueError(f"Expected a gate_analysis directory or zip file: {path}")


def choose_seed(gates: pd.DataFrame) -> int:
    seeds = sorted(int(seed) for seed in gates["seed"].unique())
    return 13 if 13 in seeds else seeds[0]


def build_figure(tables: dict[str, pd.DataFrame], seed: int) -> plt.Figure:
    gates = tables["gates"]
    correlations = tables["correlations"]
    stability = tables["stability"]
    pca = align_pc1(tables["pca"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig = plt.figure(figsize=(17, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], width_ratios=[1.2, 1.0, 1.0], hspace=0.34, wspace=0.42)

    draw_pipeline(fig.add_subplot(grid[0, 0]))
    draw_gate_heatmap(fig.add_subplot(grid[0, 1]), gates, pca, seed)
    draw_stability(fig.add_subplot(grid[0, 2]), stability)
    draw_behavior_correlations(fig.add_subplot(grid[1, :2]), correlations)
    draw_pc1_scatter(fig.add_subplot(grid[1, 2]), pca)

    fig.suptitle(
        "MoE Gate Analysis: Not Hard Reader Clusters, but a Stable Behavior-Aligned Reader Axis",
        x=0.5,
        y=0.985,
        fontsize=17,
        fontweight="bold",
    )
    return fig


def draw_pipeline(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("A. What the MoE gate does", loc="left")
    boxes = [
        (0.03, 0.58, 0.30, 0.24, "Behavior profile $z_r$\nskip / reread\nregression"),
        (0.39, 0.58, 0.22, 0.24, "Gate network\n$g(z_r)$"),
        (0.70, 0.58, 0.27, 0.24, "Expert weights\n$[w_1,w_2,w_3,w_4]$"),
        (0.39, 0.18, 0.22, 0.24, "Text encoder\nXLM-R"),
        (0.70, 0.18, 0.27, 0.24, "Weighted experts\n$\\hat{TRT}=\\sum_k w_k f_k(x)$"),
    ]
    for x, y, w, h, text in boxes:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.3,
            edgecolor="#253247",
            facecolor="#F6F8FB",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
    arrows = [
        ((0.31, 0.69), (0.38, 0.69)),
        ((0.60, 0.69), (0.69, 0.69)),
        ((0.49, 0.58), (0.49, 0.40)),
        ((0.60, 0.29), (0.69, 0.29)),
        ((0.82, 0.58), (0.82, 0.40)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.4, color="#253247"))
    ax.text(
        0.03,
        0.02,
        "Hard type assignment would make one weight near 1.\nObserved gates stay soft, but move systematically.",
        fontsize=9.2,
        color="#4A5568",
        va="bottom",
    )


def draw_gate_heatmap(ax: plt.Axes, gates: pd.DataFrame, pca: pd.DataFrame, seed: int) -> None:
    gate_cols = gate_weight_cols(gates)
    pca_seed = pca[pca["seed"] == seed][["reader_id", "pc1_aligned"]]
    subset = gates[gates["seed"] == seed].merge(pca_seed, on="reader_id").sort_values("pc1_aligned")
    matrix = subset[gate_cols].to_numpy()
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.20, vmax=0.34)
    ax.set_title(f"B. Observed gates are soft, seed {seed}", loc="left")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Readers sorted by latent axis")
    ax.set_xticks(range(len(gate_cols)))
    ax.set_xticklabels([col.replace("gate_", "E") for col in gate_cols])
    ax.set_yticks([])
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    mean_entropy = subset["gate_entropy_norm"].mean()
    mean_dom = subset["dominant_weight"].mean()
    ax.text(
        0.02,
        0.03,
        f"Mean normalized entropy = {mean_entropy:.3f}\nMean max weight = {mean_dom:.3f}\n"
        "Soft modulation, not hard clusters.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#253247",
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", boxstyle="round,pad=0.25"),
    )


def draw_stability(ax: plt.Axes, stability: pd.DataFrame) -> None:
    same = stability["same_reader_l2_t1_t2"].mean()
    different = stability["different_reader_l2_mean"].mean()
    win_rate = (stability["different_minus_same_l2"] > 0).mean()
    ax.set_title("C. Gate is reader-stable", loc="left")
    bars = ax.bar(["same reader\ntrial1 vs trial2", "different readers"], [same, different], color=["#2A9D8F", "#E76F51"])
    ax.set_ylabel("Gate L2 distance\nlower is closer")
    ax.set_ylim(0, max(different, same) * 1.45)
    for bar, value in zip(bars, [same, different]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.5,
        0.92,
        f"same < different for {win_rate:.1%} of reader rows",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#253247",
    )


def draw_behavior_correlations(ax: plt.Axes, correlations: pd.DataFrame) -> None:
    data = correlations[correlations["subset"] == "all"].copy()
    data["abs_spearman"] = data["spearman"].abs()
    summary = (
        data.groupby("behavior_feature", as_index=False)["abs_spearman"]
        .max()
        .rename(columns={"abs_spearman": "max_abs_spearman"})
        .sort_values("max_abs_spearman", ascending=True)
    )
    colors = ["#577590" if value < 0.75 else "#43AA8B" for value in summary["max_abs_spearman"]]
    ax.barh(summary["behavior_feature"], summary["max_abs_spearman"], color=colors)
    ax.set_title("D. Gate axis tracks behavior, especially rereading/regression", loc="left")
    ax.set_xlabel("Max |Spearman correlation| between gate weight and behavior feature")
    ax.set_xlim(0, 1.0)
    for idx, value in enumerate(summary["max_abs_spearman"]):
        ax.text(value + 0.015, idx, f"{value:.2f}", va="center", fontsize=9)


def draw_pc1_scatter(ax: plt.Axes, pca: pd.DataFrame) -> None:
    ax.set_title("E. PC1 is the learned reader axis", loc="left")
    for seed, group in pca.groupby("seed"):
        ax.scatter(
            group["behavior_reg.in"],
            group["pc1_aligned"],
            s=28,
            alpha=0.7,
            label=str(seed),
        )
    rho = pca[["behavior_reg.in", "pc1_aligned"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("reg.in rate in calibration trials")
    ax.set_ylabel("Aligned gate PC1")
    ax.text(0.04, 0.94, f"pooled Spearman = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
    ax.legend(title="seed", fontsize=8, title_fontsize=8, frameon=False, loc="lower right")


def align_pc1(pca: pd.DataFrame) -> pd.DataFrame:
    aligned = pca.copy()
    aligned["pc1_aligned"] = aligned["pc1"]
    for seed, group in aligned.groupby("seed"):
        corr = group[["pc1", "behavior_reg.in"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr) and corr < 0:
            aligned.loc[aligned["seed"] == seed, "pc1_aligned"] *= -1
    return aligned


def gate_weight_cols(data: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in data.columns if col.startswith("gate_") and col.split("_")[-1].isdigit()],
        key=lambda col: int(col.split("_")[-1]),
    )


if __name__ == "__main__":
    main()
