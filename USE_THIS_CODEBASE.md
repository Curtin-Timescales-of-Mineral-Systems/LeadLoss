# CDC Stability-Freeze Codebase

Canonical path:
- `/Users/lucymathieson/Peak-Picking-v2/LeadLoss/.codex-worktrees/cdc-stability-freeze`

Branch:
- `codex/cdc-stability-freeze`

Use this worktree for:
- the CDC app/reporting code that matches the current manuscript interval language
- stability-bound public exports
- the cleaned ensemble catalogue interval semantics

Do not use this worktree for:
- clustering experiments
- historical support-region reruns
- unrelated dirty refactor development

Interval semantics in this worktree:
- `ci_low` / `ci_high` = public 95% stability bounds
- `support_low` / `support_high` = geometric support span retained as diagnostics
- `ci_method` = `stability_bounds`
- `ci_interpretation` = `bootstrap_percentile_stability_bounds_of_assigned_run_ages`
