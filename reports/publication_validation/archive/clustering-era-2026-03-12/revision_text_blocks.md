# Revision text blocks (draft)

## Methods block (replace clustering/proxy description)
Discordant pre-partitioning (when enabled) is performed using an anchored age-proxy workflow. For each discordant analysis and each trial lower-intercept age in the tested grid, we evaluate candidate concordant anchor populations and compute a geometric misfit to the corresponding chord in concordia space. The discordant analysis is assigned the lower-age/anchor pair with minimum misfit, and discordant analyses are then clustered in this anchored proxy space. This replaces the previous unanchored proxy and removes circularity in per-grain proxy construction.

Peak reporting is two-stage and conservative. First, strict peaks are those that satisfy prominence/width/stability gates and are merged only when peaks are effectively unresolved at grid scale. Second, exploratory candidates (if requested) are reported separately with explicit support values and are not mixed into strict interpretations. Support is reported as Monte-Carlo reproducibility (fraction of runs voting for a peak), not as a posterior probability of geological correctness.

## Results block (benchmark summary)
Across Cases 1-7, the revised locked profile improves mean event-level absolute error in 7/7 cases relative to the manuscript profile. The largest gains occur in two-event settings (Case 2: ~95% MAE reduction; Case 5: ~74% MAE reduction; Case 7: ~53% MAE reduction). Cases 1 and 3 remain effectively unchanged at low error. Case 4 improves moderately (~17% MAE reduction) but may introduce one extra peak in one tier, indicating sensitivity in broad/complex surfaces.

Missed-event counts are reduced in the hardest two-event cases (Case 5: 3 to 0; Case 7: 3 to 1), while extra-peak counts increase slightly in Cases 4 and 5 (each +1). We therefore interpret the revised workflow as more recovery-oriented for multi-event structure, with a small trade-off of occasional over-segmentation that is controlled through strict/exploratory separation and support reporting.

## Reviewer response framing (RC1)
We agree that method validity depends on assumptions about discordance structure. We have revised the workflow and manuscript text to avoid over-strong claims and to make the assumptions operationally explicit. In particular, the discordant clustering step now uses an anchored concordant-reference proxy rather than an unanchored per-grain proxy, removing circularity in proxy assignment. We also report support as run-level reproducibility and explicitly allow unresolved outcomes (including no robust peak) when surfaces are broad or monotonic.

To test the reviewer’s concern about progressive/fanning discordance, we added an explicit fan-to-zero synthetic benchmark (new Case 8) generated with the same pipeline and plotting conventions as the existing synthetic suite. This benchmark is used as a falsification-style check: the method should not force discrete event claims when the surface does not support them.

## Reviewer response framing (RC2)
We agree that binary classifications and user-defined windows can be over-interpreted if not carefully constrained. We now clarify the exact order of operations: concordant/discordant classification is performed once on the analysis set used for a run, then Monte-Carlo perturbations are sampled around that fixed classification for run-level scoring. We also separate strict versus exploratory catalogue outputs and define support as internal reproducibility only.

In revision, we narrowed interpretive claims: CDC outputs are candidate disturbance ages that require geological corroboration, rather than automatic event determinations. We retained synthetic stress tests but reframed geologically stylized cases as diagnostics of identifiability limits, not guarantees of natural-system prevalence.

## One-paragraph cover note
We did not rewrite the core CDC architecture (reconstruction, K-S scoring, penalisation, ensemble voting); the revision focuses on guardrails and interpretability. The main technical change is anchored discordant proxying for clustering, plus strict/exploratory peak separation and explicit unresolved-state handling. This directly addresses concerns about circularity, spurious certainty, and over-interpretation while preserving reproducibility and comparability to the submitted benchmark framework.
