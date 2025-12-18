import numpy as np
import pandas as pd

df = pd.read_csv("Dec_comparison_by_event.csv")

# method labels to match your paper
method_map = {"CDC":"CDC", "Reimink":"DD", "Reimink(PEAKS)":"DD--PEAKS"}
df["Method"] = df["method"].map(method_map).fillna(df["method"])

df["Age band"] = np.where(df["true_age"] < 1000, "<1 Ga", ">=1 Ga")

out = (df.groupby(["Age band","Method"], as_index=False)
         .agg(median_bias=("bias","median"),
              mae=("abs_bias","median"),
              coverage=("covers_truth","mean")))

out["Median bias (Ma)"] = out["median_bias"].round(0).astype(int).map(lambda x: f"{x:+d}")
out["MAE (Ma)"] = out["mae"].round(0).astype(int)
out["Coverage (%)"] = (100*out["coverage"]).round(0).astype(int)

print(out[["Age band","Method","Median bias (Ma)","MAE (Ma)","Coverage (%)"]])
