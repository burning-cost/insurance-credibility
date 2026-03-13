# insurance-credibility

Credibility models for UK non-life insurance pricing: Bühlmann-Straub group credibility and Bayesian experience rating at individual policy level.

## The problem

Two problems that look similar but need different tools:

**Group credibility (schemes, large accounts):** A fleet scheme has 3 years of loss history. How much should you weight it against the market rate? Too much and you are pricing noise. Too little and you leave money on the table. The Bühlmann-Straub formula gives the optimal weight — it depends on the scheme's own variance, the portfolio variance, and the amount of exposure observed.

**Individual policy experience rating:** A commercial motor policy has been with you for 5 years with no claims. Flat NCD tables say "maximum discount". But how much is 5 years of no-claims worth relative to the a priori GLM rate? Depends on portfolio heterogeneity (how much do individual risks actually differ?), exposure (5 years at 0.5 fleet size is worth less than 5 years at 2.0), and claim frequency (low-frequency risks take longer to accumulate credible experience).

This library addresses both.

## Installation

```bash
pip install insurance-credibility
```

## Quick start

```python
# Group-level credibility (scheme pricing)
from insurance_credibility import BuhlmannStraub

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")
print(bs.credibility_factors_)
print(bs.kappa_)

# Individual policy experience rating
from insurance_credibility import ClaimsHistory, StaticCredibilityModel

histories = [
    ClaimsHistory("POL001", periods=[1,2,3], claim_counts=[0,1,0],
                  exposures=[1.0, 1.0, 0.8], prior_premium=400.0),
    ClaimsHistory("POL002", periods=[1,2,3], claim_counts=[2,1,2],
                  exposures=[1.0, 1.0, 1.0], prior_premium=400.0),
]

model = StaticCredibilityModel()
model.fit(histories)

cf = model.predict(histories[0])
posterior_premium = histories[0].prior_premium * cf
```

## Models

### Classical credibility

`BuhlmannStraub` — group credibility for scheme pricing. Estimates structural parameters (within-group variance, between-group variance) from the portfolio using method of moments. Produces credibility factors and credibility-weighted predictions per group.

`HierarchicalBuhlmannStraub` — nested group structure (e.g., scheme → book, sector → district → area). Extends Bühlmann-Straub to multi-level hierarchies following Jewell (1975).

### Experience rating

`StaticCredibilityModel` — Bühlmann-Straub at individual policy level. Fits kappa = sigma^2 / tau^2 from a portfolio of policy histories. Credibility weight for a policy is `omega = e_total / (e_total + kappa)`.

`DynamicPoissonGammaModel` — Poisson-gamma state-space model following Ahn, Jeong, Lu & Wüthrich (2023). Seniority-weighted updates: recent years count more. Produces the full posterior distribution, not just a point estimate.

`SurrogateModel` — IS-surrogate (Calcetero et al. 2024). Suitable for large portfolios where computing the exact posterior for every policy is expensive.

## Data format

```python
from insurance_credibility import ClaimsHistory

history = ClaimsHistory(
    policy_id="POL001",
    periods=[1, 2, 3, 4, 5],          # year indices
    claim_counts=[0, 1, 0, 0, 2],     # observed claims
    exposures=[1.0, 1.0, 0.8, 1.0, 1.0],  # vehicle-years
    prior_premium=450.0,               # GLM-based a priori rate
)
```

`exposures` is the key parameter that distinguishes this from flat NCD tables: a policy with 0.5 years of exposure gets far less credibility than one with 5 years, regardless of claim count.

## Performance

Benchmarked against **flat NCD table** (standard UK 5-step NCD: 0 claims → no loading, 1 claim → +20%, 2+ claims → +45%) and **simple frequency ratio** (observed frequency / portfolio mean, no credibility shrinkage) on 500 synthetic fleet/commercial policies with 3 years of history and known latent true risk (Gamma-distributed). See `notebooks/benchmark_experience.py` for full methodology.

- **Gini coefficient:** Credibility experience rating consistently produces higher Gini than flat NCD because it correctly weights exposure — 6 months of no-claims gets near-zero discount adjustment (close to a priori), while 3 full years gets substantial credit. Flat NCD treats both identically.
- **RMSE vs true risk:** Credibility shrinkage towards the prior outperforms raw frequency ratio by reducing overfitting to noisy histories. A single bad year inflates the frequency ratio but receives only partial weight under Bühlmann-Straub.
- **A/E calibration:** The max A/E deviation by predicted band is lower for credibility than for NCD, which is binned discretely and misses gradations within each claim-count band.
- **The key advantage:** correct exposure weighting. The fitted kappa determines how much exposure is needed before own experience dominates the a priori. For typical commercial motor (kappa ~ 3-8), 3 full vehicle-years gives 30-50% credibility — substantially less than flat NCD implies.
- **Limitation:** `StaticCredibilityModel` assumes homoscedastic within-policy variance. For portfolios with systematic heteroscedasticity (young drivers vs experienced, HGV vs private car), fit separately by segment. Kappa estimation needs at least 50-100 policies with 2+ years of history.

## References

- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.
- Ahn, J.Y., Jeong, H., Lu, Y. & Wüthrich, M.V. (2023). "Individual claims reserving using the Poisson-gamma state-space model." *Insurance: Mathematics and Economics*.
- Calcetero, V., Badescu, A. & Lin, X.S. (2024). "Credibility theory for the 21st century." *ASTIN Bulletin*.
- Wüthrich, M.V. (2024). "Transformer models for individual experience rating." *European Actuarial Journal*.

## Licence

MIT
