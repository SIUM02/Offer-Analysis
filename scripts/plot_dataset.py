import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import load_catalogue

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(BASE, "data", "training_profiles.csv")

THEME = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "second": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "second": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}

OPERATORS = ["Grameenphone", "Banglalink", "Teletalk", "Robi"]
CATEGORIES = ["Data", "Combo", "Minute", "SMS", "Unlimited"]


def style(theme):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["ink"],
        "axes.labelcolor": theme["second"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": theme["ink"],
        "figure.dpi": 200,
    })


def bare(ax, theme, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def caption(fig, theme, text):
    fig.text(0.012, 0.015, text, color=theme["muted"], fontsize=7.5, ha="left")


def fig_catalogue(catalogue, theme, out):
    counts = (catalogue.groupby(["operator", "category"]).size()
              .unstack(fill_value=0).reindex(OPERATORS)
              .reindex(columns=CATEGORIES, fill_value=0))

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    left = [0.0] * len(counts)
    for slot, category in enumerate(CATEGORIES):
        values = counts[category].to_numpy()
        ax.barh(counts.index, values, left=left, height=0.62,
                color=theme["series"][slot], label=category,
                edgecolor=theme["surface"], linewidth=1.4, zorder=3)
        for row, (start, value) in enumerate(zip(left, values)):
            if value >= 8:
                ax.text(start + value / 2, row, str(value), ha="center",
                        va="center", fontsize=8, color=theme["surface"],
                        fontweight="bold", zorder=4)
        left = [a + b for a, b in zip(left, values)]

    for row, total in enumerate(left):
        ax.text(total + 1.5, row, f"{total:.0f}", va="center", fontsize=9,
                color=theme["ink"], fontweight="bold")

    ax.set_title("What each operator sells")
    ax.set_xlabel("packs a subscriber can buy for data, minutes or SMS")
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.08)
    bare(ax, theme, grid_axis="x")
    ax.legend(frameon=False, ncols=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=8,
              labelcolor=theme["second"])
    caption(fig, theme, "203 buyable packs; tariff, roaming and app-only "
                        "packs excluded. Source: data/all_operators_official.csv")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_price_volume(catalogue, theme, out):
    with_data = catalogue[catalogue["data_gb"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    for slot, (operator, ax) in enumerate(zip(OPERATORS, axes.flat)):
        packs = with_data[with_data["operator"] == operator]
        ax.scatter(packs["data_gb"], packs["price_bdt"], s=42,
                   color=theme["series"][slot], alpha=0.85,
                   edgecolor=theme["surface"], linewidth=0.8, zorder=3)
        ax.set_title(f"{operator}   ({len(packs)} packs with data)",
                     fontsize=9.5, loc="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        bare(ax, theme, grid_axis="both")

    for ax in axes[1]:
        ax.set_xlabel("data in the pack (GB, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("price (BDT, log)")

    fig.suptitle("Price against volume, operator by operator", x=0.012,
                 ha="left", fontsize=11, fontweight="bold")
    caption(fig, theme, "Both axes are logarithmic: packs span 0.1-150 GB and "
                        "BDT 7-2,997. Packs with no data are not shown.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_price_per_gb(catalogue, theme, out):
    pure = catalogue[(catalogue["data_gb"] > 0) & (catalogue["minutes"] == 0)
                     & (catalogue["sms"] == 0)]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for slot, operator in enumerate(OPERATORS):
        rates = pure.loc[pure["operator"] == operator, "price_per_gb"]
        if rates.empty:
            continue
        low, mid, high = rates.quantile([0.25, 0.5, 0.75])
        colour = theme["series"][slot]
        ax.plot([low, high], [slot, slot], color=colour, linewidth=2,
                solid_capstyle="round", zorder=3)
        ax.scatter([mid], [slot], s=110, color=colour, zorder=4,
                   edgecolor=theme["surface"], linewidth=1.2)
        ax.text(high * 1.12, slot, f"median BDT {mid:,.0f}/GB   "
                f"({len(rates)} packs)", va="center", fontsize=8.5,
                color=theme["ink"])

    ax.set_yticks(range(len(OPERATORS)), OPERATORS, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("price per GB (BDT, log) — dot is the median, bar the middle half")
    ax.set_title("What a gigabyte costs")
    ax.set_xlim(3, 4000)
    ax.invert_yaxis()
    bare(ax, theme, grid_axis="x")
    caption(fig, theme, "Data-only packs, so each price buys one thing. "
                        "Grameenphone's median is dragged up by many small "
                        "content packs; its bulk rate is far lower.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_validity(catalogue, theme, out):
    days = catalogue["validity_days"]
    buckets = [("< 1 day", (days < 1).sum()), ("1", (days == 1).sum()),
               ("2-3", days.between(2, 3).sum()), ("5-7", days.between(5, 7).sum()),
               ("10-15", days.between(10, 15).sum()),
               ("28-30", days.between(28, 30).sum()),
               ("> 30", (days > 30).sum())]
    labels = [b[0] for b in buckets]
    values = [int(b[1]) for b in buckets]

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(labels, values, width=0.62, color=theme["series"][0], zorder=3)
    for x, value in enumerate(values):
        ax.text(x, value + 1.5, str(value), ha="center", fontsize=8.5,
                color=theme["ink"], fontweight="bold")
    ax.set_title("How long a pack lasts")
    ax.set_xlabel("validity")
    ax.set_ylabel("packs")
    ax.set_ylim(0, max(values) * 1.16)
    bare(ax, theme)
    caption(fig, theme, "Monthly and weekly packs dominate; the recommender "
                        "prices anything shorter as a repeat purchase.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_labels(catalogue, theme, out, top=15):
    if not os.path.exists(PROFILES):
        print(f"  (skipped label figure: no {PROFILES})")
        return
    profiles = pd.read_csv(PROFILES)
    counts = profiles["offer_id"].value_counts()
    names = catalogue.set_index("offer_id")

    head = counts.head(top)[::-1]
    labels, values = [], []
    for offer_id, count in head.items():
        row = names.loc[offer_id]
        labels.append(f"{row['offer_name'][:30]}  ({row['operator']})")
        values.append(count)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(labels, values, height=0.66, color=theme["series"][0], zorder=3)
    for row, value in enumerate(values):
        ax.text(value + max(values) * 0.012, row,
                f"{value:,}  ({value / len(profiles):.1%})", va="center",
                fontsize=8, color=theme["ink"])

    rest = len(counts) - top
    ax.set_title(f"The packs the cost model picks most often")
    ax.set_xlabel(f"times chosen across {len(profiles):,} training requests")
    ax.set_xlim(0, max(values) * 1.22)
    bare(ax, theme, grid_axis="x")
    caption(fig, theme,
            f"Top {top} of {len(counts)} packs ever chosen (of {len(catalogue)} "
            f"buyable); {rest} more share the rest. This is the label the "
            "three models are trained to reproduce.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Draws the catalogue and the training labels as figures.")
    parser.add_argument("--dark", action="store_true",
                        help="draw for a dark surface, into figures/dark/")
    args = parser.parse_args()

    theme = THEME["dark" if args.dark else "light"]
    out_dir = os.path.join(BASE, "figures", "dark" if args.dark else "")
    os.makedirs(out_dir, exist_ok=True)
    style(theme)

    catalogue = load_catalogue()
    figures = [
        ("1_catalogue.png", fig_catalogue),
        ("2_price_volume.png", fig_price_volume),
        ("3_price_per_gb.png", fig_price_per_gb),
        ("4_validity.png", fig_validity),
        ("5_labels.png", fig_labels),
    ]
    for name, draw in figures:
        path = os.path.join(out_dir, name)
        draw(catalogue, theme, path)
        if os.path.exists(path):
            print(f"  {path}  ({os.path.getsize(path) / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
