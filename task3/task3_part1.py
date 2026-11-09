import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from task3_setup import load_data

AGE_ORDER = [
    "15->19", "20->24", "25->29", "30->34", "35->39", "40->44", "45->49",
    "50->54", "55->59", "60->64", "65->69", "70->74", "75->79", "80->84", "85->89",
]

EXCLUDE_ANZSCO1 = {"Not in Work Force", "Missing/Refused"}


def build_occupation_age_table(df):
   
    df = df[~df["anzsco1"].isin(EXCLUDE_ANZSCO1)]

    df = df.drop_duplicates(subset="persid")

    counts = pd.crosstab(df["anzsco1"], df["agegroup"])

    age_columns = []
    for age in AGE_ORDER:
        if age in counts.columns:
            age_columns.append(age)
    counts = counts.reindex(columns=age_columns)

    row_totals = counts.sum(axis=1)
    row_pct = counts.div(row_totals, axis=0) * 100
    row_pct = row_pct.fillna(0.0)

    return counts, row_pct


def make_charts(counts, row_pct, out_path="part1_occupation_by_age.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    counts.T.plot(kind="bar", ax=axes[0], legend=True, width=0.85)
    axes[0].set_title("Occupation by age group -- bar chart")
    axes[0].set_xlabel("Age group")
    axes[0].set_ylabel("Number of people")
    axes[0].legend(fontsize=6, loc="upper right")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)

    im = axes[1].imshow(row_pct.values, aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(row_pct.columns)))
    axes[1].set_xticklabels(row_pct.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(row_pct.index)))
    axes[1].set_yticklabels(row_pct.index, fontsize=8)
    axes[1].set_title("Occupation by age group -- heatmap")
    fig.colorbar(im, ax=axes[1], label="% of occupation")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_part1_summary(counts, row_pct, path="part1_occupation_by_age_summary.json"):
    row_sums = row_pct.sum(axis=1)
    spot_check = None
    if "Sales Workers" in row_pct.index and "15->19" in row_pct.columns:
        spot_check = row_pct.loc["Sales Workers", "15->19"]

    payload = {
        "counts_total": str(int(counts.values.sum())),
        "occupation_categories": sorted(str(c) for c in counts.index),
        "age_group_categories": sorted(str(c) for c in counts.columns),
        "row_pct_row_sums": {
            str(occ): f"{v:.10f}" for occ, v in row_sums.to_dict().items()
        },
        "row_pct_min": f"{row_pct.values.min():.10f}",
        "row_pct_max": f"{row_pct.values.max():.10f}",
        "spot_check_sales_workers_15_19": (
            f"{spot_check:.10f}" if spot_check is not None else None
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def part1():
    df = load_data()
    counts, row_pct = build_occupation_age_table(df)
    make_charts(counts, row_pct)
    _save_part1_summary(counts, row_pct)
    return counts, row_pct

