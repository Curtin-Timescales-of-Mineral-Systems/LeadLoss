#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from process import calculations  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    return p.parents[2]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _resolve_inputs(prefix: Path) -> dict[str, Path]:
    prefix = prefix.expanduser().resolve()
    stem = prefix.with_suffix("") if prefix.suffix else prefix
    return {
        "summary": stem.parent / f"{stem.name}_summary.csv",
        "anchors": stem.parent / f"{stem.name}_anchors.csv",
        "spots": stem.parent / f"{stem.name}_spots.csv",
        "clusters": stem.parent / f"{stem.name}_clusters.csv",
        "peaks": stem.parent / f"{stem.name}_peaks.csv",
        "rejected": stem.parent / f"{stem.name}_rejected.csv",
    }


def _concordia_curve(age_max_ma: float = 4000.0, n: int = 1200):
    ages_y = np.linspace(1.0e6, float(age_max_ma) * 1.0e6, int(n))
    x = np.array([calculations.u238pb206_from_age(t) for t in ages_y], float)
    y = np.array([calculations.pb207pb206_from_age(t) for t in ages_y], float)
    return ages_y / 1.0e6, x, y


def _cluster_color(cid: int) -> str:
    cmap = plt.get_cmap("tab10")
    return cmap(int(cid) % 10)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def make_figure(
    summary_df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    spots_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    sample_id: str,
):
    srow = summary_df.loc[summary_df["sample"].astype(str) == str(sample_id)]
    if srow.empty:
        raise ValueError(f"Sample {sample_id} not found in diagnostics summary.")
    srow = srow.iloc[0]

    anchors = anchors_df.loc[anchors_df["sample"].astype(str) == str(sample_id)].copy()
    spots = spots_df.loc[spots_df["sample"].astype(str) == str(sample_id)].copy()
    clusters = clusters_df.loc[clusters_df["sample"].astype(str) == str(sample_id)].copy()
    peaks = peaks_df.loc[peaks_df["sample"].astype(str) == str(sample_id)].copy()
    rejected = rejected_df.loc[rejected_df["sample"].astype(str) == str(sample_id)].copy()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    conc = spots.loc[spots["role"] == "concordant"].copy()
    conc_ages = pd.to_numeric(conc["reference_age_ma"], errors="coerce").dropna().to_numpy(float)
    if conc_ages.size:
        bins = min(max(6, conc_ages.size // 2), 18)
        ax_a.hist(conc_ages, bins=bins, color="0.85", edgecolor="0.25", lw=0.6)
        y_top = ax_a.get_ylim()[1]
        for _, row in anchors.iterrows():
            age = pd.to_numeric(row["anchor_age_ma"], errors="coerce")
            if pd.isna(age):
                continue
            ax_a.axvline(float(age), color=_cluster_color(int(row["anchor_id"])), lw=1.4)
            ax_a.text(float(age), y_top * 0.96, f"A{int(row['anchor_id'])}", rotation=90,
                      va="top", ha="right", fontsize=8)
        ax_a.set_xlabel("Concordant age (Ma)")
        ax_a.set_ylabel("Count")
    else:
        ax_a.text(0.5, 0.5, "No concordant ages", ha="center", va="center", transform=ax_a.transAxes)
        ax_a.set_axis_off()
    ax_a.set_title("Concordant anchors")

    ages_ma, cx, cy = _concordia_curve()
    ax_b.plot(cx, cy, color="tab:blue", lw=1.0, zorder=1)
    for label_age in [500, 1000, 1500, 2000, 2500, 3000]:
        x = calculations.u238pb206_from_age(label_age * 1.0e6)
        y = calculations.pb207pb206_from_age(label_age * 1.0e6)
        ax_b.plot([x], [y], "o", ms=2.5, color="tab:blue", zorder=2)
        ax_b.text(x, y, f" {label_age}", fontsize=7, va="bottom")

    conc = spots.loc[spots["role"] == "concordant"].copy()
    disc = spots.loc[spots["role"] == "discordant"].copy()
    rev = spots.loc[spots["role"] == "reverse_discordant"].copy()

    if not conc.empty:
        ax_b.scatter(
            pd.to_numeric(conc["u238u206pb"], errors="coerce"),
            pd.to_numeric(conc["pb207pb206"], errors="coerce"),
            s=18,
            c="mediumseagreen",
            label="Concordant",
            zorder=3,
        )

    if not disc.empty:
        ambiguous_mask = disc["ambiguous"].astype(str).str.lower().isin(["true", "1"])
        amb = disc.loc[ambiguous_mask]
        for cid, grp in disc.loc[~ambiguous_mask].groupby("cluster_id", dropna=True):
            if str(cid) == "" or pd.isna(cid):
                continue
            ax_b.scatter(
                pd.to_numeric(grp["u238u206pb"], errors="coerce"),
                pd.to_numeric(grp["pb207pb206"], errors="coerce"),
                s=22,
                c=[_cluster_color(int(float(cid)))],
                label=f"Cluster {int(float(cid))}",
                zorder=4,
            )
        if not amb.empty:
            ax_b.scatter(
                pd.to_numeric(amb["u238u206pb"], errors="coerce"),
                pd.to_numeric(amb["pb207pb206"], errors="coerce"),
                s=22,
                c="0.5",
                marker="x",
                label="Ambiguous",
                zorder=4,
            )

    if not rev.empty:
        ax_b.scatter(
            pd.to_numeric(rev["u238u206pb"], errors="coerce"),
            pd.to_numeric(rev["pb207pb206"], errors="coerce"),
            s=20,
            c="crimson",
            marker="+",
            label="Reverse discordant",
            zorder=4,
        )

    ax_b.set_xlabel(r"$^{238}$U/$^{206}$Pb")
    ax_b.set_ylabel(r"$^{207}$Pb/$^{206}$Pb")
    ax_b.set_title("TW plot + assignments")
    ax_b.legend(loc="upper right", frameon=False)

    ax_c.axis("off")
    lines = [
        f"Sample: {sample_id}",
        f"Clustering requested: {_as_bool(srow['clustering_requested'])}",
        f"Split accepted: {_as_bool(srow['split_accepted'])}",
        f"Reporting accepted: {_as_bool(srow['reporting_accepted'])}",
        f"Reason: {str(srow['reason']) or '—'}",
        f"Anchors: {int(srow['n_anchors']) if pd.notna(srow['n_anchors']) else 0}",
        f"Discordant assigned/ambiguous: "
        f"{int(srow['n_assigned']) if pd.notna(srow['n_assigned']) else 0}/"
        f"{int(srow['n_ambiguous']) if pd.notna(srow['n_ambiguous']) else 0}",
        "",
        "Cluster outcomes:",
    ]

    if clusters.empty:
        lines.append("  none")
    else:
        for _, row in clusters.sort_values(["cluster_id", "result_type"]).iterrows():
            cid = int(row["cluster_id"])
            n = int(row["cluster_n"]) if pd.notna(row["cluster_n"]) else 0
            proxy = pd.to_numeric(row["cluster_proxy_median_ma"], errors="coerce")
            result_type = str(row["result_type"])
            if result_type == "resolved_peak":
                lines.append(
                    f"  C{cid}: n={n}, proxy={proxy:.1f} Ma, peak={float(row['age_ma']):.1f} Ma "
                    f"[{float(row['ci_low_ma']):.1f}, {float(row['ci_high_ma']):.1f}]"
                )
            elif result_type == "recent_boundary_mode":
                lines.append(
                    f"  C{cid}: n={n}, proxy={proxy:.1f} Ma, recent boundary mode <= {float(row['upper_bound_ma']):.1f} Ma"
                )
            elif result_type == "rejected":
                lines.append(
                    f"  C{cid}: n={n}, proxy={proxy:.1f} Ma, rejected ({row['reason']})"
                )
            else:
                lines.append(f"  C{cid}: n={n}, proxy={proxy:.1f} Ma, no reportable peak")

    if not peaks.empty:
        lines.extend(["", "Reported rows:"])
        for _, row in peaks.iterrows():
            mode = str(row.get("mode", ""))
            if mode == "recent_boundary":
                lines.append(
                    f"  recent boundary mode <= {float(row['ci_high_ma']):.1f} Ma "
                    f"(direct {100*float(row['direct_support']):.0f}%)"
                )
            else:
                lines.append(
                    f"  {float(row['age_ma']):.1f} Ma "
                    f"(direct {100*float(row['direct_support']):.0f}%, winner {100*float(row['winner_support']):.0f}%)"
                )

    if not rejected.empty:
        lines.extend(["", "Rejected rows:"])
        for _, row in rejected.iterrows():
            age = pd.to_numeric(row["age_ma"], errors="coerce")
            age_txt = f"{float(age):.1f} Ma" if pd.notna(age) else "n/a"
            lines.append(f"  {age_txt}: {row['reason']}")

    ax_c.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5)
    ax_c.set_title("Cluster outcome summary")

    return fig


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()
    ap = argparse.ArgumentParser(description="Plot clustering diagnostics for one sample.")
    ap.add_argument("--prefix", type=Path, required=True, help="Base path used by the clustering diagnostics export.")
    ap.add_argument("--sample-id", type=str, required=True, help="Sample identifier to plot.")
    ap.add_argument("--fig-dir", type=Path, default=(paper_dir / "outputs" / "figures"))
    ap.add_argument("--outfile-stub", type=str, default=None)
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _resolve_inputs(args.prefix)
    summary_df = _load_csv(paths["summary"])
    anchors_df = _load_csv(paths["anchors"])
    spots_df = _load_csv(paths["spots"])
    clusters_df = _load_csv(paths["clusters"])
    peaks_df = _load_csv(paths["peaks"])
    rejected_df = _load_csv(paths["rejected"])

    fig = make_figure(
        summary_df,
        anchors_df,
        spots_df,
        clusters_df,
        peaks_df,
        rejected_df,
        args.sample_id,
    )

    if not args.no_save:
        outdir = args.fig_dir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        stub = args.outfile_stub or f"fig_clustering_diagnostics_{args.sample_id}"
        for ext in [x.strip() for x in args.formats.split(",") if x.strip()]:
            out = outdir / f"{stub}.{ext}"
            fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
            print("[saved]", out)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
