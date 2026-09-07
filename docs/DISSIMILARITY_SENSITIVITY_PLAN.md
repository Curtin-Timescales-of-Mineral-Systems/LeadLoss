# Dissimilarity-measure sensitivity plan

LeadLoss currently uses the two-sample Kolmogorov-Smirnov (KS) statistic. A
different statistic can change the shape and scale of the age-search surface,
so adding a radio button is not enough to make another method scientifically
interchangeable with KS.

## Sensible first comparison

Use Cramér-von Mises as the first alternative sensitivity measure because it
compares the whole empirical cumulative distribution rather than only the
largest separation. Wasserstein distance can be included as a second,
descriptive comparison, but its units and any normalisation must be declared
explicitly. Keep both outside the production GUI until the tests below pass.

## Required checks

For KS and each candidate measure, use identical simulated analyses and random
draws, then compare:

- bias and recovery rate for known injected Pb-loss ages;
- false positive or spurious-peak rate when no resolvable event is present;
- sensitivity to concordant and discordant analysis counts;
- sensitivity to analytical uncertainty and Wetherill rho;
- sensitivity to unimodal, mixed and boundary-limited cases;
- TW-versus-Wetherill agreement;
- changes in ensemble-peak, broad-best-fit and unresolved classifications.

The ensemble thresholds must then be calibrated for that measure. A threshold
tuned to a KS-based 0–1 statistic should not automatically be reused for a
statistic with a different sampling distribution or an arbitrary rescaling.

## Decision rule

Promote an alternative measure into the GUI only if it has a documented score
definition, deterministic tests, acceptable recovery and false-positive
behaviour, and a clear explanation of whether it is a primary method or a
sensitivity check. Until then, KS remains the manuscript and production
default.
