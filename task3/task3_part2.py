import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from task3_setup import load_data

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOURS = list(range(4, 24))


def build_hour_table(df):
   
    df = df[df["starthour"] <= 23]

    ct = pd.crosstab(df["travdow"], df["starthour"])
    ct = ct.reindex(index=DOW_ORDER, columns=HOURS, fill_value=0)
    
    marginal = ct.sum(axis=0)

    return ct, marginal


def make_charts(ct, marginal, out_path="part2_trips_by_hour.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(HOURS, marginal.values, color="#2980b9")
    axes[0].set_title("Trips by hour of day -- bar chart")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Total trips")

    colors = plt.cm.tab10(range(7))
    for i, d in enumerate(DOW_ORDER):
        axes[1].plot(HOURS, ct.loc[d].values, label=d, color=colors[i], alpha=0.8)
    axes[1].set_title("Trips by hour of day -- line chart")
    axes[1].set_xlabel("Hour of day")
    axes[1].legend(fontsize=7, ncol=2)

    im = axes[2].imshow(ct.values, aspect="auto", cmap="YlOrRd")
    axes[2].set_yticks(range(7))
    axes[2].set_yticklabels(DOW_ORDER)
    axes[2].set_xticks(range(len(HOURS)))
    axes[2].set_xticklabels(HOURS, fontsize=7)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_title("Trips by hour of day -- heatmap")
    fig.colorbar(im, ax=axes[2], label="Total trips", fraction=0.03)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_part2_summary(ct, marginal, path="part2_trips_by_hour_summary.json"):
    try:
        marginal_matches_ct_sum = bool(marginal.equals(ct.sum(axis=0)))
    except Exception:
        marginal_matches_ct_sum = False

    marginal_at_8 = None
    if 8 in marginal.index:
        marginal_at_8 = str(int(marginal[8]))

    payload = {
        "ct_total": str(int(ct.values.sum())),
        "travdow_categories": sorted(str(c) for c in ct.index),
        "hour_categories": sorted(str(c) for c in ct.columns),
        "has_null": bool(ct.isnull().values.any()),
        "ct_min": str(int(ct.values.min())),
        "ct_max": str(int(ct.values.max())),
        "marginal_matches_ct_sum": marginal_matches_ct_sum,
        "marginal_at_8": marginal_at_8,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def part2():
    df = load_data()
    ct, marginal = build_hour_table(df)
    make_charts(ct, marginal)
    _save_part2_summary(ct, marginal)
    return ct, marginal

