"""
Poisson-Gamma conjugate Bayesian credibility model.

This module implements the Poisson-Gamma conjugate credibility model — the
closed-form Bayesian update for claim frequency data. No MCMC required. No new
dependencies beyond scipy (already in the package).

The model is the exact Bayesian answer to the question: "Given a Gamma prior on
a group's Poisson rate, and some observed claims, what is the posterior
distribution of the rate?" The conjugacy of the Gamma-Poisson pair makes this
analytically tractable: the posterior is another Gamma, the credibility estimate
is the posterior mean, and credibility intervals come directly from Gamma quantiles.

Mathematical background
-----------------------
Each group i is modelled as:

    Level 1 (likelihood):   claims_{ij} | lambda_i ~ Poisson(exposure_{ij} * lambda_i)
    Level 2 (prior):        lambda_i ~ Gamma(alpha, beta)

where lambda_i is the group's underlying claim rate (claims per unit exposure).

The Gamma prior has:
    - mean: alpha / beta  (= portfolio mean rate)
    - variance: alpha / beta^2

After observing sum_j claims_{ij} claims over sum_j exposure_{ij} total exposure,
the posterior is:

    lambda_i | data ~ Gamma(alpha + N_i, beta + E_i)

where N_i = sum_j claims_{ij} and E_i = sum_j exposure_{ij}.

This gives:
    Posterior mean:      mu_i = (alpha + N_i) / (beta + E_i)
    Credibility factor:  Z_i  = E_i / (E_i + beta)
    Prior mean:          mu_0 = alpha / beta

The credibility premium is exactly the posterior mean:

    mu_i = Z_i * (N_i / E_i) + (1 - Z_i) * mu_0

This is structurally identical to Bühlmann credibility but with a rigorous
Bayesian derivation. The "k parameter" is beta — the prior effective exposure.

Prior calibration
-----------------
The prior parameters alpha and beta are estimated from portfolio data via
method-of-moments on group-level observed rates. Specifically, we match:

    E[lambda] = alpha / beta = weighted mean of observed rates
    Var[lambda] = alpha / beta^2 = weighted variance of observed rates

This gives:
    beta  = mean_rate / var_rate
    alpha = mean_rate * beta

When the observed variance is very small (groups are near-homogeneous), beta
will be large, reflecting a tightly concentrated prior — the model will produce
low credibility factors and pull all groups toward the portfolio mean.

References
----------
Klugman, S.A., Panjer, H.H. & Willmot, G.E. (2012). Loss Models: From Data to
    Decisions (4th ed.). Wiley. Chapter 20: Credibility.
Bühlmann, H. & Gisler, A. (2005). A Course in Credibility Theory and its
    Applications. Springer. Chapter 2: The Poisson-Gamma model.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import polars as pl
from scipy.stats import gamma as scipy_gamma


class PoissonGammaCredibility:
    """
    Poisson-Gamma conjugate Bayesian credibility model for claim frequency.

    This is the closed-form Bayesian credibility model for claim count data.
    Each group has an unknown claim rate lambda_i; we place a Gamma prior on
    lambda_i calibrated from portfolio data, and the posterior after observing
    the group's own experience is another Gamma.

    The credibility estimate is the posterior mean — a weighted average of the
    group's observed rate and the portfolio prior mean. The weights are
    determined entirely by the group's exposure relative to the prior's
    effective exposure (beta).

    This model requires claim counts (not rates) and exposures as separate
    inputs. If you only have pre-computed loss ratios, use BuhlmannStraub.

    Parameters
    ----------
    prior_alpha : float, optional
        Shape parameter of the Gamma prior on lambda. If None, estimated from
        data using method-of-moments. Providing a prior directly bypasses
        the data-driven calibration — useful when you have strong external
        beliefs about the portfolio mean and variance.
    prior_beta : float, optional
        Rate parameter of the Gamma prior on lambda. If None, estimated from
        data. Together with alpha: prior mean = alpha/beta, prior variance =
        alpha/beta^2.
    min_prior_variance : float, default 1e-8
        Floor on the estimated between-group rate variance during calibration.
        When all groups show the same observed rate (zero variance), the
        method-of-moments estimate of beta diverges. This floor prevents
        beta from becoming numerically infinite. In practice, a very large
        beta means Z_i ≈ 0 for all groups — they all get the prior mean.

    Examples
    --------
    >>> import polars as pl
    >>> from insurance_credibility.classical import PoissonGammaCredibility
    >>>
    >>> df = pl.DataFrame({
    ...     "scheme":   ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    ...     "year":     [2021, 2022, 2023] * 3,
    ...     "claims":   [45, 52, 48, 12, 15, 11, 120, 130, 125],
    ...     "exposure": [1000, 1100, 1050, 300, 320, 290, 5000, 4800, 5200],
    ... })
    >>> model = PoissonGammaCredibility()
    >>> model.fit(df, group_col="scheme", claims_col="claims", exposure_col="exposure")
    >>> model.summary()
    """

    def __init__(
        self,
        prior_alpha: Optional[float] = None,
        prior_beta: Optional[float] = None,
        min_prior_variance: float = 1e-8,
    ) -> None:
        if prior_alpha is not None and prior_alpha <= 0:
            raise ValueError(f"prior_alpha must be positive, got {prior_alpha}")
        if prior_beta is not None and prior_beta <= 0:
            raise ValueError(f"prior_beta must be positive, got {prior_beta}")
        if (prior_alpha is None) != (prior_beta is None):
            raise ValueError(
                "prior_alpha and prior_beta must both be provided or both be None. "
                "Supplying only one is ambiguous. Provide both to use an external "
                "prior, or omit both to calibrate from data."
            )

        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_prior_variance = min_prior_variance

        # Fitted attributes — set by .fit()
        self._alpha: Optional[float] = None
        self._beta: Optional[float] = None
        self._premiums: Optional[pl.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Union[pl.DataFrame, "pd.DataFrame"],  # type: ignore[name-defined]
        group_col: str = "group",
        claims_col: str = "claims",
        exposure_col: str = "exposure",
        period_col: Optional[str] = None,
    ) -> "PoissonGammaCredibility":
        """
        Fit the Poisson-Gamma model to panel (or cross-sectional) claim data.

        Parameters
        ----------
        data:
            A Polars DataFrame (preferred) or pandas DataFrame. One row per
            (group, period) for panel data, or one row per group for aggregate
            data. For panel data, all periods are pooled: N_i = sum of claims,
            E_i = sum of exposures.
        group_col:
            Column identifying the group (scheme, territory, NCD class, etc.).
        claims_col:
            Column with claim counts (non-negative integers or floats). Must
            be counts, not rates. If you have pre-aggregated totals across
            all periods, use one row per group.
        exposure_col:
            Column with exposure (earned car years, policy count, etc.). Must
            be strictly positive.
        period_col:
            Optional column identifying the time period. Only used for
            validation (duplicate detection). Not required for fitting.

        Returns
        -------
        self
            Returns the fitted estimator so calls can be chained.
        """
        from ._validation import _to_polars
        data = _to_polars(data)

        self._validate_inputs(data, group_col, claims_col, exposure_col, period_col)

        # Aggregate to group level: N_i, E_i, observed rate r_i = N_i / E_i
        groups = (
            data.group_by(group_col)
            .agg([
                pl.col(claims_col).sum().alias("N_i"),
                pl.col(exposure_col).sum().alias("E_i"),
            ])
            .rename({group_col: "group"})
            .sort("group")
        )
        groups = groups.with_columns(
            (pl.col("N_i") / pl.col("E_i")).alias("r_i")
        )

        N = groups["N_i"].to_numpy().astype(float)
        E = groups["E_i"].to_numpy().astype(float)
        r = groups["r_i"].to_numpy()
        group_ids = groups["group"].to_list()

        # Calibrate or use provided prior parameters
        if self.prior_alpha is not None and self.prior_beta is not None:
            alpha = float(self.prior_alpha)
            beta = float(self.prior_beta)
        else:
            alpha, beta = self._calibrate_prior(N, E, r)

        # Posterior parameters for each group
        alpha_post = alpha + N          # shape: alpha + N_i
        beta_post = beta + E            # rate: beta + E_i

        # Posterior mean (= credibility premium)
        mu_post = alpha_post / beta_post

        # Credibility factor: Z_i = E_i / (E_i + beta)
        z = E / (E + beta)

        # Prior mean
        mu_prior = alpha / beta

        # Store fitted state
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._groups = groups
        self._group_ids = group_ids

        self._premiums = pl.DataFrame({
            "group": group_ids,
            "total_claims": N,
            "total_exposure": E,
            "observed_rate": r,
            "Z": z,
            "credibility_rate": mu_post,
            "prior_mean": np.full(len(groups), mu_prior),
            "posterior_alpha": alpha_post,
            "posterior_beta": beta_post,
        })

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict (new groups or new data)
    # ------------------------------------------------------------------

    def predict(
        self,
        claims: float,
        exposure: float,
        credibility_interval: float = 0.95,
    ) -> dict:
        """
        Compute the posterior credibility estimate for a new group.

        Given a new group's observed claims and exposure, applies the fitted
        prior to produce a posterior estimate. This is the main prediction
        entry point for pricing a scheme not seen during fitting.

        Parameters
        ----------
        claims:
            Total claim count for the new group.
        exposure:
            Total exposure for the new group.
        credibility_interval:
            Width of the credibility interval (default 0.95 = 95%).

        Returns
        -------
        dict with keys:
            ``credibility_rate`` : posterior mean
            ``Z``                : credibility factor
            ``lower``            : lower bound of credibility interval
            ``upper``            : upper bound of credibility interval
            ``posterior_alpha``  : Gamma posterior shape
            ``posterior_beta``   : Gamma posterior rate
        """
        self._check_fitted()
        if claims < 0:
            raise ValueError(f"claims must be non-negative, got {claims}")
        if exposure <= 0:
            raise ValueError(f"exposure must be positive, got {exposure}")

        alpha_post = self._alpha + claims
        beta_post = self._beta + exposure
        mu_post = alpha_post / beta_post
        z = exposure / (exposure + self._beta)

        tail = (1.0 - credibility_interval) / 2.0
        lower = scipy_gamma.ppf(tail, a=alpha_post, scale=1.0 / beta_post)
        upper = scipy_gamma.ppf(1.0 - tail, a=alpha_post, scale=1.0 / beta_post)

        return {
            "credibility_rate": float(mu_post),
            "Z": float(z),
            "lower": float(lower),
            "upper": float(upper),
            "posterior_alpha": float(alpha_post),
            "posterior_beta": float(beta_post),
        }

    # ------------------------------------------------------------------
    # Credibility intervals for fitted groups
    # ------------------------------------------------------------------

    def credibility_intervals(
        self,
        credibility_interval: float = 0.95,
    ) -> pl.DataFrame:
        """
        Return posterior credibility intervals for all fitted groups.

        Uses the Gamma posterior distribution directly — no asymptotic
        approximation, no bootstrapping. The intervals are exact under the
        Poisson-Gamma model.

        Parameters
        ----------
        credibility_interval:
            Width of the interval. 0.95 gives 2.5th–97.5th percentiles.

        Returns
        -------
        pl.DataFrame with columns:
            ``group``, ``credibility_rate``, ``lower``, ``upper``, ``Z``
        """
        self._check_fitted()

        if not (0 < credibility_interval < 1):
            raise ValueError(
                f"credibility_interval must be in (0, 1), got {credibility_interval}"
            )

        tail = (1.0 - credibility_interval) / 2.0
        alpha_post = self._premiums["posterior_alpha"].to_numpy()
        beta_post = self._premiums["posterior_beta"].to_numpy()

        lower = scipy_gamma.ppf(tail, a=alpha_post, scale=1.0 / beta_post)
        upper = scipy_gamma.ppf(1.0 - tail, a=alpha_post, scale=1.0 / beta_post)

        return pl.DataFrame({
            "group": self._group_ids,
            "credibility_rate": self._premiums["credibility_rate"].to_numpy(),
            "lower": lower,
            "upper": upper,
            "Z": self._premiums["Z"].to_numpy(),
        })

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha_(self) -> float:
        """
        Fitted prior shape parameter (alpha).

        Prior mean = alpha / beta. A larger alpha (with fixed beta) means a
        stronger, more concentrated prior.
        """
        self._check_fitted()
        return self._alpha  # type: ignore[return-value]

    @property
    def beta_(self) -> float:
        """
        Fitted prior rate parameter (beta).

        This is the 'effective prior exposure' — a group needs exposure equal
        to beta to achieve Z = 0.5. Analogous to Bühlmann's k parameter.
        """
        self._check_fitted()
        return self._beta  # type: ignore[return-value]

    @property
    def prior_mean_(self) -> float:
        """Portfolio prior mean claim rate (alpha / beta)."""
        self._check_fitted()
        return self._alpha / self._beta  # type: ignore[operator]

    @property
    def premiums_(self) -> pl.DataFrame:
        """
        Credibility estimates — one row per group.

        Columns:

        - ``group``            : group identifier
        - ``total_claims``     : observed total claim count N_i
        - ``total_exposure``   : observed total exposure E_i
        - ``observed_rate``    : empirical rate N_i / E_i
        - ``Z``                : credibility factor E_i / (E_i + beta)
        - ``credibility_rate`` : posterior mean (alpha + N_i) / (beta + E_i)
        - ``prior_mean``       : portfolio prior mean alpha / beta
        - ``posterior_alpha``  : Gamma posterior shape parameter
        - ``posterior_beta``   : Gamma posterior rate parameter
        """
        self._check_fitted()
        return self._premiums  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pl.DataFrame:
        """
        Print a formatted model summary and return the credibility estimates.

        Structural parameters (prior alpha, beta, prior mean) are printed to
        stdout. The per-group credibility table is returned as a Polars
        DataFrame for downstream use.

        Returns
        -------
        pl.DataFrame
            Per-group credibility estimates.
        """
        self._check_fitted()

        print("Poisson-Gamma Credibility Model")
        print("=" * 46)
        print(f"  Prior shape     alpha = {self._alpha:.6g}")
        print(f"  Prior rate      beta  = {self._beta:.6g}   (effective prior exposure)")
        print(f"  Prior mean      mu_0  = {self.prior_mean_:.6g}   (alpha / beta)")
        print()
        print(
            "  Interpretation: a group needs exposure = beta to achieve Z = 0.50"
        )
        print(
            "  Credibility formula: rate = Z * observed_rate + (1-Z) * prior_mean"
        )
        print()

        tbl = self._premiums.select([
            "group",
            "total_exposure",
            "observed_rate",
            "Z",
            "credibility_rate",
            "prior_mean",
        ]).rename({
            "total_exposure": "Exposure",
            "observed_rate": "Obs. Rate",
            "credibility_rate": "Cred. Rate",
            "prior_mean": "Prior Mean",
        })
        return tbl

    # ------------------------------------------------------------------
    # Prior calibration
    # ------------------------------------------------------------------

    def _calibrate_prior(
        self,
        N: np.ndarray,
        E: np.ndarray,
        r: np.ndarray,
    ) -> tuple[float, float]:
        """
        Estimate Gamma prior parameters via method-of-moments on group rates.

        We match the first two moments of the Gamma distribution to the
        exposure-weighted mean and variance of group-level observed rates:

            E[lambda]   = alpha / beta  = weighted mean of r_i
            Var[lambda] = alpha / beta^2 = weighted variance of r_i

        Solving gives:
            beta  = mean_rate / var_rate
            alpha = mean_rate * beta = mean_rate^2 / var_rate

        The variance estimate is weighted by E_i (group total exposure), so
        larger groups have more influence on the prior calibration. This is the
        natural weighting: groups with more data are more informative about the
        portfolio distribution.

        If the estimated variance is near zero (groups appear homogeneous),
        beta is floored at mean_rate / min_prior_variance to avoid infinity.
        A warning is issued in this case.
        """
        E_total = E.sum()
        w = E / E_total  # normalised exposure weights

        mean_rate = float((w * r).sum())

        # Exposure-weighted variance: sum_i w_i * (r_i - mean_r)^2
        var_rate = float((w * (r - mean_rate) ** 2).sum())

        if var_rate < self.min_prior_variance:
            warnings.warn(
                f"Estimated between-group rate variance ({var_rate:.3e}) is very small. "
                "Groups appear near-homogeneous. "
                f"Flooring variance at min_prior_variance={self.min_prior_variance:.3e}. "
                "This will produce a high-beta prior (large effective prior exposure) "
                "and Z near zero for all groups — they all get the portfolio mean. "
                "Consider supplying prior_alpha and prior_beta directly if you have "
                "external information about the portfolio's heterogeneity.",
                stacklevel=4,
            )
            var_rate = self.min_prior_variance

        beta = mean_rate / var_rate
        alpha = mean_rate * beta

        return float(alpha), float(beta)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        data: pl.DataFrame,
        group_col: str,
        claims_col: str,
        exposure_col: str,
        period_col: Optional[str],
    ) -> None:
        required = {group_col, claims_col, exposure_col}
        if period_col is not None:
            required.add(period_col)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        if data.is_empty():
            raise ValueError("data is empty.")

        for col in [group_col, claims_col, exposure_col]:
            n_null = data[col].null_count()
            if n_null > 0:
                raise ValueError(
                    f"Column '{col}' contains {n_null} null value(s). "
                    "Remove or impute before fitting."
                )

        if (data[claims_col].cast(pl.Float64) < 0).any():
            raise ValueError(
                f"Column '{claims_col}' contains negative values. "
                "Claims must be non-negative counts."
            )

        if (data[exposure_col].cast(pl.Float64) <= 0).any():
            raise ValueError(
                f"Column '{exposure_col}' contains non-positive values. "
                "Exposure must be strictly positive."
            )

        n_groups = data[group_col].n_unique()
        if n_groups < 2:
            raise ValueError(
                "At least 2 groups are required to calibrate the prior. "
                f"Found {n_groups} group. "
                "If you have only one group, supply prior_alpha and prior_beta directly."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )

    def __repr__(self) -> str:
        if not self._fitted:
            return "PoissonGammaCredibility(not fitted)"
        return (
            f"PoissonGammaCredibility("
            f"alpha={self._alpha:.4g}, "
            f"beta={self._beta:.4g}, "
            f"prior_mean={self.prior_mean_:.4g})"
        )
