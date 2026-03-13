# Ensemble-Only Manuscript Cut Map

Target manuscript:
[/Users/lucymathieson/Downloads/template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex)

This map identifies the algorithmic clustering references that should be removed or replaced if the manuscript is aligned to the ensemble-only CDC branch. Geological uses of the word `cluster` are not included unless they could be confused with the algorithm.

## Do Not Touch

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L147)
  This refers to geological nano-sized Pb clusters, not the CDC algorithm.

## Replace

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L155)
  Replace the paragraph beginning `Here we directly address...`.
  Use the replacement block in [ensemble_only_methods_blocks.tex](/Users/lucymathieson/Peak-Picking-v2/LeadLoss/reports/publication_validation/ensemble_only_methods_blocks.tex).

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L198)
  Replace the full `CDC enhancements` opening paragraph so it no longer lists clustering as one of the three advances.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L282)
  Replace the opening of `Per--run peak detection` so it defines the per-run raw and penalised goodness curves directly, without any `across-cluster minima` language.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L989)
  Replace the runtime sentence so it refers to `per-run peak detection and the ensemble catalogue`, not `across--cluster minima`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1095)
  Replace the paragraph beginning `The new workflow differs in three ways.` so the third point is `explicit unresolved outcomes`, not clustering.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1109)
  Replace the sentence listing the discussion topics. Remove `clustering`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1169)
  Replace the entire subsubsection `Clustering acceptance and fallback` with `Conservative abstention and boundary handling`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1226)
  Replace the `Splayed-chord geometry` paragraph to remove the sentence claiming clustering helps in that geometry.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1231)
  Replace the opening conclusions paragraph so it describes the ensemble-only workflow.

## Delete Entirely

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L211)
  Delete the entire subsection `Discordant analysis clustering`.
  This runs from the subsection heading through the fallback sentence at line 278.

## Delete Or Rewrite In Place

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L198)
  Remove the cross-reference `Sect.~\ref{sec:disc-analysis-cluster}`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L283)
  Remove `After clustering`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L285)
  Remove `across-cluster minima`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1095)
  Remove `when discordant clustering is accepted` and `Sect.~\ref{sec:disc-analysis-cluster}`.

- [template_revision.tex](/Users/lucymathieson/Downloads/template_revision.tex#L1177)
  Remove `single cluster comparison`.

## Internal Consistency Checks After Editing

- Remove any remaining `\label{sec:disc-analysis-cluster}` references.
- Recompile and confirm there are no broken `\ref{sec:disc-analysis-cluster}` references.
- Search the manuscript for:
  - `clustering`
  - `clustered`
  - `across-cluster`
  - `single cluster`
  - `disc-analysis-cluster`
- Leave geological `clusters` references intact unless they clearly refer to the algorithm.
