# reimink_cdc_runtime_summary.py
import pandas as pd, numpy as np, pathlib as pl

# --- EDIT THESE PATHS ---
CDC_LOG = pl.Path("/Users/lucymathieson/Desktop/LeadLossOutputs/runtime_log_28Nov.csv")
DD_LOG  = pl.Path("/Users/lucymathieson/Desktop/reimink_discordance_dating/runtime_log_reimink.csv")

def load_any(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalise tier to uppercase if present
    if "tier" in df.columns:
        df["tier"] = (
            df["tier"]
              .astype(str)
              .str.strip()
              .str.upper()
        )

    # infer tier if missing (e.g. from sample names '1A', '2B', …)
    if "tier" not in df.columns and "sample" in df.columns:
        df["tier"] = (
            df["sample"].astype(str)
              .str.extract(r"([A-Za-z])$", expand=False)
              .str.upper()
        )
    return df

def summarise_cdc(df):
    e2e = df.query("method.str.upper()=='CDC'", engine="python")
    e2e = e2e[e2e["phase"].str.lower().eq("e2e_runtime")]
    per = df.query("method.str.upper()=='CDC' and phase.str.upper()=='MC'", engine="python")

    by_tier = (
        e2e.groupby("tier", dropna=False)["elapsed_s"]
           .agg(median="median", min="min", max="max")
           .round(2)
           .assign(per_run_median_s = per.groupby("tier")["per_run_median_s"].median().round(3))
           .reset_index()
           .sort_values("tier")
    )
    overall = dict(
        e2e_med = round(e2e["elapsed_s"].median(), 1),
        per_run = round(per["per_run_median_s"].median(), 3)
    )
    return by_tier, overall

def summarise_dd(df):
    """
    Summarise the Reimink/DD runtime log with columns:
    timestamp, case, tier, sample, step, elapsed_sec, status, note
    """
    required = {"timestamp", "case", "sample", "tier", "step", "elapsed_sec"}
    if not required.issubset(df.columns):
        raise ValueError(f"summarise_dd expects columns {required}, got {sorted(df.columns)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # sort so runs are in order
    df = df.sort_values(["case", "sample", "timestamp"])

    # assign a run_id within each (case, sample) every time we see script_start
    df["run_id"] = (
        df["step"].eq("script_start")
          .groupby([df["case"], df["sample"]])
          .cumsum()
    )

    # drop anything before the first script_start
    df = df[df["run_id"] > 0]

    # End-to-end: rows where step == script_end; elapsed_sec is full runtime
    e2e = df[df["step"].eq("script_end")].copy()
    e2e = e2e.rename(columns={"elapsed_sec": "elapsed_s"})

    # Per-bootstrap median per run: only bootstrap_* steps
    boot = df[df["step"].str.startswith("bootstrap_")]
    per_boot_run = (
        boot.groupby(["case", "sample", "run_id"])["elapsed_sec"]
            .median()
            .reset_index(name="per_boot_median_s")
    )

    # attach per-run bootstrap medians to the e2e rows
    e2e = e2e.merge(per_boot_run, on=["case", "sample", "run_id"], how="left")

    # ---- by tier ----
    by_tier = (
        e2e.groupby("tier", dropna=False)["elapsed_s"]
           .agg(median="median", min="min", max="max")
           .round(2)
           .reset_index()
           .sort_values("tier")
    )

    # median per-bootstrap time per tier
    per_boot_tier = (
        e2e.groupby("tier")["per_boot_median_s"]
           .median()
           .round(2)
    )
    by_tier["per_boot_median_s"] = by_tier["tier"].map(per_boot_tier)

    # ---- overall medians ----
    overall = dict(
        e2e_med = round(e2e["elapsed_s"].median(), 1),
        per_boot = None if e2e["per_boot_median_s"].isna().all()
                       else round(float(np.nanmedian(e2e["per_boot_median_s"])), 2)
    )
    return by_tier, overall

def add_speedup(cdc_tbl, dd_tbl):
    m = pd.merge(cdc_tbl, dd_tbl, on="tier", suffixes=("_cdc", "_dd"))
    # CDC is faster → speedup = Reimink / CDC
    m["speedup_x"] = (m["median_dd"] / m["median_cdc"]).round(2)

    cols = [
        "tier",
        "median_cdc",
        "median_dd",
        "per_run_median_s",
        "per_boot_median_s",
        "speedup_x",          # <-- moved to the end
    ]

    return m[cols].rename(columns={
        "median_cdc": "CDC E2E median (s)",
        "median_dd": "Reimink E2E median (s)",
        "per_run_median_s": "CDC per-run (s)",
        "per_boot_median_s": "Reimink per-boot (s)",
        "speedup_x": "CDC speedup vs Reimink (×)",   # clearer label
    })

cdc_tbl, cdc_over = summarise_cdc(load_any(CDC_LOG))
dd_tbl,  dd_over  = summarise_dd(load_any(DD_LOG))
cmp_tbl = add_speedup(cdc_tbl, dd_tbl)

print("\nCDC (by tier)\n", cdc_tbl.to_string(index=False))
print("\nReimink/DD (by tier)\n", dd_tbl.to_string(index=False))
print("\nComparison (speedup)\n", cmp_tbl.to_string(index=False))
print("\nOverall medians:",
      f"\n  CDC E2E median  = {cdc_over['e2e_med']} s; CDC per‑run = {cdc_over['per_run']} s",
      f"\n  DD  E2E median  = {dd_over['e2e_med']} s; DD per‑boot = {dd_over['per_boot']} s")
