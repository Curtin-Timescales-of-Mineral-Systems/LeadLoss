## Freeze Checklist

This repository now has two analysis branches:

- `codex/ensemble-only`
  - path: `/Users/lucymathieson/Peak-Picking-v2/LeadLoss`
  - role: no-clustering manuscript baseline
- `codex/clustering-from-ensemble`
  - path: `/Users/lucymathieson/Peak-Picking-v2/LeadLoss/.codex-worktrees/clustering-from-ensemble`
  - role: same baseline plus clustering extension

These branches should be treated as `source-only frozen` analysis candidates.
Do not treat existing derived outputs as frozen until they are regenerated from
the locked source state.

## Lock Before Reruns

Before running any benchmark or natural-case suite, lock the following choices
and do not change them sample-by-sample:

- clustering branch to use
- concordance sigma rule (`1σ` or `2σ`)
- modelling window and node count
- Monte Carlo run count
- clustering toggle
- nearby-peak merge setting
- boundary-mode reporting rule

If natural clustered analyses are run at `1σ`, record that as the main natural
clustering setting and treat `2σ` as sensitivity analysis only.

## Gatekeeper Reruns

Run these first before any large batch:

- synthetic `2A`
- synthetic `3A`
- synthetic `4C`
- synthetic `8A`
- natural `96969025`
- natural `96969042`
- natural `97969138`

Only proceed to full reruns if these look correct.

## Full Locked Reruns

After the gatekeeper pass succeeds, rerun:

- synthetic `Cases 1-4`
- synthetic `Case 8 fan-to-zero`
- any remaining synthetic cases still discussed in the manuscript
- the short list of natural examples that may appear in the paper

## Reproducibility Packaging

After reruns complete and outputs are accepted:

- regenerate tables
- regenerate figures
- export clustering diagnostics where used
- archive old derived outputs if needed
- make one local snapshot commit or release note for the rerun state

## Manuscript Work Only After Freeze

Only after the rerun outputs are locked:

- update figure panels and captions
- update methods text
- update results tables
- update discussion and reviewer responses
