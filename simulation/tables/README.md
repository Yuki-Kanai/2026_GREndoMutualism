# tables/ — units and column conventions

All growth rates (`mu_S`, `mu_H`, `mu_H_0`, `delta_mu_H`) are in the model's own units:
`mu = R_m w`, bounded above by `1^T c^E = 1` (`01_model_spec.md` §10-3).  Pre-evolution
`|c^H|_1 = 0.01`, so pre-evolution `mu_S <= 0.01`.  Counts (`n_bounded`, `n_unbounded`,
`ub`, `n_excreted`, `r_plus`, `n_degenerate`) are dimensionless integers; `alpha = sqrt(b)`
and `s = ||S||_F` are dimensionless.  Times are seconds.

Every summarised column appears as `<name>_<statistic>` with statistics
`n`, `n_nan`, `mean`, `median`, `trimmed_mean_5`, `sd`, `q10`, `q25`, `q75`, `q90`,
`min`, `max`.  `<name>_ci_lo` / `<name>_ci_hi` are percentile bootstrap 95 % intervals of
the **mean**, resampled at the level of `symbiont_id` (B = 2000, seeds derived from the
master seed through the `boot` namespace).

| file | one row is | notes |
|---|---|---|
| `R1_stats.csv` | one `r` | pre-evolution `mu_S`, `mu_H`, `delta_mu_H`; saturation and infeasibility rates; within-`r` Spearman |
| `R1_correlations.csv` | one statistic | pooled / partial / within-`r` Spearman of `(mu_H, mu_S)` |
| `R2_stats.csv` | one `r` | `s`, `alpha`, `b`, `u`, `ub`, `n_excreted`, `r_plus`, `n_degenerate`, `cond_b` |
| `R2_peaks.csv` | one (quantity, statistic) | `peak_at` with bootstrap CI, LOESS peak, the mean `b` at the peak, and the peak-minus-endpoint contrasts |
| `R2_curve_<q>_<stat>.csv` | one `r` | the bootstrapped curve the peak was taken from |
| `R3_per_symbiont.csv` | one `R^S` | `SD_b` / `SD_u` in four definitions (see below) |
| `R3_stats.csv` | one (subset, x, y, definition) | OLS slope / intercept / `R^2` / residual SD / Spearman |
| `pre_per_symbiont.csv` | one `R^S` | pre-evolution summaries used to join onto the post-evolution table |
| `R4_stats.csv` | one (`r`, subset) | subset is `all` or `no_detlimit` |
| `R4_peak.csv` | one (subset, statistic) | peak of mean / median post-evolution `mu_H` and the endpoint contrasts |
| `R4_curve.csv` | one `r` | the curve the figure is drawn from |
| `R4_correlations.csv` | one (figure, subset) | Spearman pooled / partial on `r` / within-`r` Fisher, plus quartile contrast, Cliff's delta and the Hodges-Lehmann shift |
| `G3_milp_bench.csv` | one MILP | solve time, status, box usage, LP re-solve cross-check |
| `V10_tolerance_sensitivity.csv` | one variant | fraction of cells whose integer quantities changed, and max abs/rel differences |
| `V10_milp_presolve.csv` | one MILP | `mu_H` with CPLEX presolve off vs on |

## `SD_b` / `SD_u` definitions (R3)

Primary (`SD_b`, `SD_u`): root of the summed per-component variance of `f^S*` over the
`N_c` values of `c^H`, with the bounded set taken by majority vote (a component counts as
bounded if it was bounded in more than half of that symbiont's cells).

Secondary, all reported alongside:
`*_sequential` — per-component variance computed only over the cells where that component
was actually bounded / unbounded, weighted by how often it was;
`*_pooled_scalar` — the scalar SD over the pooled (cell x component) sample;
`*_mean_component` — the mean of the per-component SDs.

## Raw-value policy

`mu_S` and `mu_H` are stored raw.  Nothing is clipped, rounded or zeroed.  Level 2
infeasibility is `mu_H = NaN` plus `flag_infeasible`; the paper's post-hoc conventions are
derived in the statistics scripts, never stored.

## Columns added in run 2 (not yet in `05_conventions.md` §4.1)

The cell-level schema gained six columns.  **They are reported here so the
author can decide whether to fold them into the spec; the implementation is
already emitting them.**

| column | stage | meaning |
|---|---|---|
| `n_offered` | both | `#{i : c^H_i > tol_abs}`.  Diagnostic for the post-evolution `b` (A19 / N1 §2). **Never a replacement for `b`.** |
| `n_weakly_active` | both | `#{i : c^H_i = 0 and λ¹_i = 0}`.  Constraints that are active only weakly. |
| `mip_best_bound` | post | CPLEX best bound at termination, clipped to ≤ 1.  Needed to bound a `detlimit` cell by an interval instead of re-solving it (07 §2.3). |
| `n_rxn_effective` | both | Number of distinct reactions appearing in the columns of `R^S`.  The effective metabolic repertoire — the direct evidence for reviewer point M2 (A1). |
| `mode_size_mean` / `_min` / `_max` | both | Reactions per mode.  Constant `= k` for `paper` / `k_sweep`; varies as `n_rxn/r` for `fixed_repertoire`, where `k` is undefined and the `k` column is null. |

`n_rxn` was already in the schema but was written as null in run 1; it now carries
the fixed-repertoire size.

## A1 tables (gate G2a)

| file | one row per | contents |
|---|---|---|
| `A1_repertoire_size.csv` | arm × `r` | effective repertoire, mode sizes, mean `b`, mean `ub`, support |
| `A1_curves.csv` | arm × quantity × statistic × axis × group | curve value with bootstrap 95 % CI and support |
| `A1_peaks.csv` | arm × quantity × statistic × axis | peak position, its bootstrap CI, and the contrasts against both endpoints |

`axis` is `r` or `b`.  On the `b` axis, groups supported by fewer than 20 symbionts
are dropped before the bootstrap (a handful of symbionts cannot support one).
