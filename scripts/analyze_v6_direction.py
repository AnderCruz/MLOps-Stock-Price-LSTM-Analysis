from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


N = 180
ACCURACY = 0.5698

CORRECT = round(N * ACCURACY)

result = binomtest(
    CORRECT,
    N,
    p=0.5,
    alternative="greater",
)

print("=" * 70)
print("V6 DIRECTIONAL SIGNAL ANALYSIS")
print("=" * 70)

print(f"Samples:       {N}")
print(f"Correct:       {CORRECT}")
print(f"Accuracy:      {ACCURACY:.4%}")
print(f"Baseline:      50.0000%")
print(f"Improvement:   {(ACCURACY - 0.50) * 100:.2f} pp")

print()
print("Binomial test")
print(f"p-value:       {result.pvalue:.6f}")

ci = result.proportion_ci(
    confidence_level=0.95,
)

print()
print("95% CI")
print(f"Lower:         {ci.low:.4%}")
print(f"Upper:         {ci.high:.4%}")

print()
print("=" * 70)

if result.pvalue < 0.05:
    print("STATISTICAL SIGNAL: PASS")
else:
    print("STATISTICAL SIGNAL: INCONCLUSIVE")

print("=" * 70)
