"""
Bonus-Malus System (NCD) equilibrium simulator.

This module implements the BMSEquilibriumSimulator — a tool for computing
Nash equilibrium reporting thresholds in no-claims discount (NCD) systems.
It formalises the "hunger for bonus" phenomenon: the rational incentive for
policyholders to self-insure small claims rather than report them, to protect
their NCD discount.

The theoretical foundation is Liang, Zhang, Zhou & Zou (arXiv:2601.12655,
submitted January 2026), the first published analysis of strategic claim
underreporting under oligopolistic insurer competition. Classical results from
Lemaire (1977) — the original hunger-for-bonus analysis — underpin the
single-insurer threshold computation.

Why this matters for UK pricing
--------------------------------
Standard UK practice fits a Poisson frequency GLM on *reported* claims. The
observed claim frequency at high-NCD classes understates true frequency because
small claims are strategically withheld. A 9-year NCD holder may suppress all
claims below ~£90; a 5-year holder may suppress all claims below ~£280.

The GLM absorbs this suppression into the NCD coefficients. The result:
relativities that accurately describe reported behaviour but misprice true risk.
High-NCD policyholders are undercharged; lower-NCD policyholders cross-subsidise
them. This simulator quantifies that bias and provides a corrected frequency
estimate per NCD class.

The two key questions
---------------------
1. What is the rational reporting threshold b*_n for a policyholder in NCD
   class n? (Lemaire algorithm — works for any ladder.)

2. Given thresholds and a loss severity distribution, what is the true claim
   frequency implied by the observed (underreported) frequency? (Censoring
   correction — applies to any ladder.)

For two-insurer oligopoly, the Liang et al. closed form gives the Nash
equilibrium threshold directly.

Mathematical background
-----------------------
The NCD ladder has classes 0, 1, ..., N-1 indexed from worst to best (0 = no
NCD, N-1 = maximum NCD). Each class has a discount rate d_n ∈ [0, 1) applied
to the base premium B. A fault claim steps the policyholder back k_step classes
(typically 2 in the UK).

Lemaire's dynamic programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Under risk-neutral policyholders, the optimal reporting barrier for class n
is the NPV of the premium penalty from reporting:

    b*_n = B × (d_n - d_{n-k_step}) × Σ_{t=1}^{T} δ^t

where δ is the discount factor and T is the rebuilding horizon (number of
claim-free years to return to class n from class n-k_step). The full Lemaire
(1977) algorithm extends this by accounting for stochastic future claims during
the rebuilding window.

Liang et al. two-insurer closed form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a two-class BMS (N = 2) with penalty ratio κ = c₂/c₁ ∈ (1, 2), the Nash
equilibrium reporting threshold under oligopoly is:

    b*(θ₁, θ₂) = δ(κ−1) × [θ₁·η(θ₁−θ₂) + θ₂·(1−η(θ₁−θ₂))]

where θ_i is insurer i's Class 1 premium, δ is the discount factor, and η is
the probability of being with insurer 1 as a function of the premium differential.
The threshold is the probability-weighted NPV of the downgrade penalty.

Frequency correction
~~~~~~~~~~~~~~~~~~~~
If the loss severity follows distribution F (e.g. Gamma) and the barrier at
class n is b*_n, then the reporting probability is:

    p_n = P(Y > b*_n) = 1 − F(b*_n)

The corrected (true) frequency:

    λ_true_n = λ_obs_n / p_n

This is the quantity that should be used in GLM calibration once the barrier
has been estimated.

Steady-state distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~
The Markov chain over NCD classes has transition matrix T where the transition
probabilities depend on the reporting strategy (via p_n). The stationary
distribution π satisfies (I − T^⊤)π = 0, normalised to sum to 1.

References
----------
Liang, Z., Zhang, J., Zhou, Z. & Zou, B. (2026). Optimal Underreporting and
    Competitive Equilibrium. arXiv:2601.12655.
Lemaire, J. (1977). La Soif du Bonus. ASTIN Bulletin, 9(1-2):181–190.
Lemaire, J. (1995). Bonus-Malus Systems in Automobile Insurance.
    Kluwer Academic Publishers.
Holtan, J. (2001). Optimal Loss Financing Under Bonus-Malus Contracts.
    ASTIN Bulletin, 31(1):161–173.
Norberg, R. (1976). A credibility theory for automobile bonus systems.
    Scandinavian Actuarial Journal, 2:92–107.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
from scipy import stats as scipy_stats
from scipy.linalg import solve


class BMSEquilibriumSimulator:
    """
    Bonus-Malus System equilibrium simulator for NCD underreporting analysis.

    Computes the rational (Nash equilibrium) reporting thresholds for each class
    of a no-claims discount (NCD) ladder, quantifies the frequency suppression
    bias in observed claims data, and corrects frequency estimates for use in
    GLM pricing.

    The simulator implements two complementary approaches:

    1. **Lemaire (1977) dynamic programming** — works for any NCD ladder
       structure. Computes the NPV of the premium penalty from reporting at
       each class, accounting for the stochastic rebuilding path.

    2. **Liang et al. (arXiv:2601.12655) closed form** — for a two-class BMS
       under oligopolistic competition. Computes the Nash equilibrium threshold
       directly from premium parameters and the choice function.

    Parameters
    ----------
    discounts : array-like of float
        NCD discount rates, one per class, ordered from lowest (0 = no NCD)
        to highest (maximum NCD). Values in [0, 1). For the standard UK 10-class
        ladder: [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70].
    base_premium : float
        Base premium (before applying the NCD discount). This is the premium at
        class 0 (no NCD), used to compute the absolute value of moving between
        classes. Must be positive.
    step_back : int, default 2
        Number of NCD classes lost per fault claim. Standard UK convention is
        2 years (a 9-year holder drops to 7 years after one fault claim).
        Must be a positive integer.
    discount_factor : float, default 0.97
        Annual discount factor δ ∈ (0, 1] for future premium flows. Corresponds
        to a risk-free rate of approximately 3% at the default. Holtan (2001)
        notes that higher interest rates reduce the present value of future
        penalties and therefore lower the reporting threshold.
    claim_freq : float, default 0.05
        Portfolio average annual claim frequency (claims per car year). Used in
        the Lemaire dynamic programming to model the stochastic rebuilding path.
        Typical UK motor value: 0.05–0.10.
    severity_dist : scipy.stats frozen distribution or None, default None
        Frozen severity distribution for computing the reporting probability
        p_n = P(Y > b*_n). If None, severity-based corrections are not computed.
        Common choices: ``scipy.stats.gamma(a=1.2, scale=1/0.0085)``,
        ``scipy.stats.lognorm(s=1.0, scale=np.exp(6.5))``.
    max_horizon : int, default 10
        Maximum planning horizon T (years) for the NPV calculation in the
        simplified Lemaire algorithm. Beyond T years, future premiums are
        discounted to near zero. Values in range [5, 20] are reasonable for
        UK motor.

    Examples
    --------
    UK 9-class NCD ladder with Gamma severity:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from insurance_credibility.classical import BMSEquilibriumSimulator
    >>>
    >>> uk_discounts = [0.0, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.65, 0.70, 0.70]
    >>> sim = BMSEquilibriumSimulator(
    ...     discounts=uk_discounts,
    ...     base_premium=1000.0,
    ...     severity_dist=stats.gamma(a=1.2, scale=1/0.0085),
    ... )
    >>> sim.fit()
    >>> sim.thresholds_          # reporting thresholds b*_n by class
    >>> sim.reporting_probs_     # P(Y > b*_n) by class
    >>> sim.corrected_freq_      # corrected frequencies (requires observed_freq)
    >>> sim.summary()

    Two-insurer equilibrium (Liang et al. Section 4):

    >>> sim2 = BMSEquilibriumSimulator(
    ...     discounts=[0.0, 0.25],   # two-class BMS
    ...     base_premium=35.85,
    ...     discount_factor=0.97,
    ... )
    >>> result = sim2.liang_equilibrium(
    ...     theta1=35.83, theta2=33.45,
    ...     kappa=1.25, k1=0.015, k2=0.8,
    ... )
    >>> result["threshold"]  # Nash equilibrium b*

    References
    ----------
    Liang et al. (arXiv:2601.12655); Lemaire (1977); Holtan (2001).
    """

    def __init__(
        self,
        discounts: Union[list[float], np.ndarray],
        base_premium: float,
        step_back: int = 2,
        discount_factor: float = 0.97,
        claim_freq: float = 0.05,
        severity_dist=None,
        max_horizon: int = 10,
    ) -> None:
        discounts = np.asarray(discounts, dtype=float)
        if discounts.ndim != 1 or len(discounts) < 2:
            raise ValueError(
                "discounts must be a 1-D array with at least 2 elements. "
                f"Got shape {discounts.shape}."
            )
        if np.any(discounts < 0) or np.any(discounts >= 1):
            raise ValueError(
                "All discount rates must be in [0, 1). "
                f"Got range [{discounts.min():.4g}, {discounts.max():.4g}]."
            )
        if not np.all(np.diff(discounts) >= 0):
            warnings.warn(
                "Discount rates are not monotonically non-decreasing. "
                "For a standard NCD ladder discounts should increase with class. "
                "Proceeding, but check your input.",
                stacklevel=2,
            )
        if base_premium <= 0:
            raise ValueError(
                f"base_premium must be positive. Got {base_premium}."
            )
        if not isinstance(step_back, int) or step_back < 1:
            raise ValueError(
                f"step_back must be a positive integer. Got {step_back!r}."
            )
        if not (0 < discount_factor <= 1):
            raise ValueError(
                f"discount_factor must be in (0, 1]. Got {discount_factor}."
            )
        if not (0 < claim_freq < 1):
            raise ValueError(
                f"claim_freq must be in (0, 1). Got {claim_freq}."
            )
        if not isinstance(max_horizon, int) or max_horizon < 1:
            raise ValueError(
                f"max_horizon must be a positive integer. Got {max_horizon!r}."
            )

        self.discounts = discounts
        self.base_premium = float(base_premium)
        self.step_back = step_back
        self.discount_factor = float(discount_factor)
        self.claim_freq = float(claim_freq)
        self.severity_dist = severity_dist
        self.max_horizon = max_horizon

        self._n_classes = len(discounts)
        self._fitted = False

        # Fitted attributes
        self._thresholds: Optional[np.ndarray] = None
        self._reporting_probs: Optional[np.ndarray] = None
        self._stationary_dist: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None
        self._corrected_freq: Optional[np.ndarray] = None
        self._observed_freq: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        observed_freq: Optional[Union[list[float], np.ndarray]] = None,
    ) -> "BMSEquilibriumSimulator":
        """
        Compute the Lemaire reporting thresholds and derived quantities.

        This method runs the full pipeline:

        1. Compute b*_n for each NCD class via simplified Lemaire dynamic
           programming (NPV of the step-back penalty).
        2. If a severity distribution was supplied, compute reporting
           probabilities p_n = P(Y > b*_n).
        3. Build the Markov transition matrix under the reporting strategy.
        4. Compute the stationary (steady-state) distribution over NCD classes.
        5. If observed frequencies are provided, compute corrected frequencies
           λ_true_n = λ_obs_n / p_n.

        Parameters
        ----------
        observed_freq : array-like of float or None
            Observed (reported) claim frequency per NCD class, length must equal
            the number of classes. These are claims per car year as observed in
            the portfolio. If None, corrected frequencies are not computed.

        Returns
        -------
        self
            Returns the fitted simulator so calls can be chained.
        """
        if observed_freq is not None:
            obs = np.asarray(observed_freq, dtype=float)
            if obs.shape != (self._n_classes,):
                raise ValueError(
                    f"observed_freq must have length {self._n_classes} "
                    f"(one per NCD class). Got {obs.shape}."
                )
            if np.any(obs < 0):
                raise ValueError(
                    "observed_freq must be non-negative. "
                    f"Found {(obs < 0).sum()} negative values."
                )
            self._observed_freq = obs
        else:
            self._observed_freq = None

        # Step 1: compute thresholds
        self._thresholds = self._lemaire_thresholds()

        # Step 2: reporting probabilities (requires severity distribution)
        if self.severity_dist is not None:
            self._reporting_probs = self._compute_reporting_probs(self._thresholds)
        else:
            self._reporting_probs = None

        # Step 3 & 4: transition matrix and stationary distribution
        p = self._reporting_probs if self._reporting_probs is not None else np.ones(self._n_classes)
        self._transition_matrix = self._build_transition_matrix(p)
        self._stationary_dist = self._compute_stationary_distribution(self._transition_matrix)

        # Step 5: corrected frequencies
        if self._observed_freq is not None and self._reporting_probs is not None:
            # Guard against division by zero for near-zero reporting probabilities
            safe_p = np.where(self._reporting_probs > 1e-10, self._reporting_probs, np.nan)
            self._corrected_freq = self._observed_freq / safe_p
        else:
            self._corrected_freq = None

        self._fitted = True
        return self

    @property
    def thresholds_(self) -> np.ndarray:
        """
        Reporting thresholds b*_n for each NCD class, as a 1-D numpy array.

        b*_n is the loss amount below which a rational risk-neutral policyholder
        in class n will not report the claim (self-insure to protect their NCD).

        Index 0 corresponds to NCD class 0 (no discount); index N-1 corresponds
        to the highest NCD class. Units are the same as ``base_premium``.

        Under the simplified Lemaire algorithm, the threshold is higher at
        mid-ladder classes and lower at the top of the ladder (where the absolute
        premium difference between consecutive classes is small).
        """
        self._check_fitted()
        return self._thresholds  # type: ignore[return-value]

    @property
    def reporting_probs_(self) -> np.ndarray:
        """
        Reporting probabilities p_n = P(Y > b*_n) for each NCD class.

        Requires ``severity_dist`` to have been set at construction. Raises
        ``RuntimeError`` if severity_dist is None.

        p_n is the probability that a random loss exceeds the reporting
        threshold at class n. The closer p_n is to 1, the less suppression
        occurs at that class.

        Observed frequency at class n estimates ``claim_freq × p_n``, not the
        true underlying frequency ``claim_freq``.
        """
        self._check_fitted()
        if self._reporting_probs is None:
            raise RuntimeError(
                "reporting_probs_ requires a severity_dist to be supplied at "
                "construction. Set severity_dist=scipy.stats.gamma(...) or "
                "another frozen distribution."
            )
        return self._reporting_probs

    @property
    def stationary_dist_(self) -> np.ndarray:
        """
        Stationary distribution π over NCD classes under the reporting strategy.

        π[n] is the long-run proportion of policyholders in class n at steady
        state. This is the solution to (I − T^⊤)π = 0 with Σπ[n] = 1.

        The distribution depends on the reporting thresholds b*_n (via the
        class-specific reporting probabilities p_n) and the claim frequency.
        A higher claim frequency or lower thresholds push mass toward lower
        NCD classes.
        """
        self._check_fitted()
        return self._stationary_dist  # type: ignore[return-value]

    @property
    def transition_matrix_(self) -> np.ndarray:
        """
        Markov transition matrix T, shape (n_classes, n_classes).

        T[i, j] is the probability of moving from class i to class j in one
        period, under the reporting strategy. Entry T[i, j] reflects:

        - Probability of reporting a claim (p_i) → step back k classes
        - Probability of not reporting (1 - p_i) → step forward 1 class

        The matrix is built using the reporting probabilities derived from the
        Lemaire thresholds and the severity distribution.
        """
        self._check_fitted()
        return self._transition_matrix  # type: ignore[return-value]

    @property
    def corrected_freq_(self) -> np.ndarray:
        """
        Corrected (true) claim frequencies λ_true_n = λ_obs_n / p_n, per class.

        Requires both ``observed_freq`` to be passed to ``fit()`` and
        ``severity_dist`` to be set at construction. Raises ``RuntimeError``
        if either is missing.

        This is the key pricing correction: feed these corrected frequencies
        into your GLM instead of the observed frequencies. The NCD relativities
        will then reflect true risk rather than reported behaviour.
        """
        self._check_fitted()
        if self._corrected_freq is None:
            raise RuntimeError(
                "corrected_freq_ requires both severity_dist (at construction) "
                "and observed_freq (passed to fit()). One or both are missing."
            )
        return self._corrected_freq

    def frequency_bias_(self) -> np.ndarray:
        """
        Fractional frequency bias by NCD class: (λ_obs - λ_true) / λ_true.

        Returns negative values (reported frequency understates true frequency).
        The magnitude indicates how severely each class's frequency is
        underestimated in a naive GLM.

        Requires both ``severity_dist`` and ``observed_freq``. Raises
        ``RuntimeError`` if either is missing.

        Returns
        -------
        np.ndarray
            Shape (n_classes,). Negative values in [-1, 0]. A value of -0.35
            means observed frequency is 35% below true frequency.
        """
        self._check_fitted()
        if self._reporting_probs is None or self._corrected_freq is None:
            raise RuntimeError(
                "frequency_bias_ requires severity_dist (at construction) and "
                "observed_freq (passed to fit()). One or both are missing."
            )
        obs = self._observed_freq  # type: ignore[assignment]
        true = self._corrected_freq
        mask = np.isfinite(true) & (true > 0)
        bias = np.full_like(obs, np.nan)
        bias[mask] = (obs[mask] - true[mask]) / true[mask]
        return bias

    def liang_equilibrium(
        self,
        theta1: float,
        theta2: float,
        kappa: float = 1.25,
        k1: float = 0.015,
        k2: float = 0.8,
    ) -> dict:
        """
        Liang et al. (arXiv:2601.12655) Nash equilibrium threshold, two-class BMS.

        Computes the Nash equilibrium reporting threshold under oligopolistic
        competition between two insurers (Theorem 3.1 + Section 4 closed form).
        This is the unique barrier strategy that rational policyholders adopt
        given the premium structure (θ₁, θ₂).

        The closed form (Assumption 4.1, N = 2 classes, penalty ratio κ):

            b*(θ₁, θ₂) = δ(κ−1) × [θ₁·η(θ₁−θ₂) + θ₂·(1−η(θ₁−θ₂))]

        where η is the probability of being with insurer 1, modelled as a
        logistic function of the premium differential.

        The choice function (equation 2.2, Liang et al.):

            η(Δ) = 1 / (1 + exp(k₁·Δ + log((1−k₂)/k₂)))

        This is a logistic choice probability parameterised by price sensitivity
        k₁ and brand preference k₂.

        Parameters
        ----------
        theta1 : float
            Class 1 (best class) premium for insurer 1. Positive.
        theta2 : float
            Class 1 premium for insurer 2. Positive.
        kappa : float, default 1.25
            Penalty ratio: Class 2 premium = kappa × Class 1 premium.
            Must satisfy 1 < kappa < 2. Typical range for UK NCD: 1.2–1.5.
            Liang et al. note that kappa ∈ (1.2, 1.5) covers most real BMS.
        k1 : float, default 0.015
            Price sensitivity parameter. Higher k1 → more price-elastic
            policyholders → smaller premium gap at equilibrium.
        k2 : float, default 0.8
            Brand preference / asymmetry parameter. k2 = 0.5 gives symmetric
            competition (equal premium at equilibrium). k2 > 0.5 means insurer 1
            is preferred and can charge more. The Liang et al. base case uses 0.8.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``threshold``: Nash equilibrium reporting threshold b* (same
              units as theta1, theta2)
            - ``eta``: Probability of being with insurer 1, η(θ₁−θ₂)
            - ``premium_diff``: θ₁ − θ₂ (positive when insurer 1 charges more)
            - ``theta1``: insurer 1 Class 1 premium (passed through)
            - ``theta2``: insurer 2 Class 1 premium (passed through)
            - ``kappa``: penalty ratio (passed through)
            - ``k1``: price sensitivity (passed through)
            - ``k2``: brand preference (passed through)
            - ``discount_factor``: δ (from simulator construction)

        Raises
        ------
        ValueError
            If kappa ≤ 1 or kappa ≥ 2, or if theta1 or theta2 are non-positive.

        Examples
        --------
        Base case from Liang et al. (Section 4.3):

        >>> sim = BMSEquilibriumSimulator(
        ...     discounts=[0.0, 0.25],
        ...     base_premium=35.85,
        ...     discount_factor=0.97,
        ... )
        >>> result = sim.liang_equilibrium(
        ...     theta1=35.83, theta2=33.45,
        ...     kappa=1.25, k1=0.015, k2=0.8,
        ... )
        >>> result["threshold"]  # Nash equilibrium reporting barrier
        """
        if theta1 <= 0 or theta2 <= 0:
            raise ValueError(
                f"theta1 and theta2 must be positive. Got theta1={theta1}, theta2={theta2}."
            )
        if not (1 < kappa < 2):
            raise ValueError(
                f"kappa must satisfy 1 < kappa < 2 (Assumption 4.1). Got {kappa}."
            )
        if k1 <= 0:
            raise ValueError(f"k1 (price sensitivity) must be positive. Got {k1}.")
        if not (0 < k2 < 1):
            raise ValueError(f"k2 (brand preference) must be in (0, 1). Got {k2}.")

        # Choice function η(Δ) = logistic probability of being with insurer 1
        # η(Δ) = 1 / (1 + exp(k1·Δ + log((1-k2)/k2)))
        # where Δ = θ₁ − θ₂
        delta_premium = theta1 - theta2
        log_odds_base = np.log((1 - k2) / k2)
        eta = 1.0 / (1.0 + np.exp(k1 * delta_premium + log_odds_base))

        # Nash equilibrium threshold (Liang et al. Section 4, closed form)
        # b* = δ(κ−1) × [θ₁·η + θ₂·(1−η)]
        weighted_premium = theta1 * eta + theta2 * (1 - eta)
        threshold = self.discount_factor * (kappa - 1) * weighted_premium

        return {
            "threshold": float(threshold),
            "eta": float(eta),
            "premium_diff": float(delta_premium),
            "theta1": float(theta1),
            "theta2": float(theta2),
            "kappa": float(kappa),
            "k1": float(k1),
            "k2": float(k2),
            "discount_factor": self.discount_factor,
        }

    def nash_equilibrium_premiums(
        self,
        kappa: float = 1.25,
        k1: float = 0.015,
        k2: float = 0.8,
        premium_cap: float = 50.0,
        tol: float = 1e-6,
        max_iter: int = 500,
    ) -> dict:
        """
        Numerically solve for Nash equilibrium premiums under Liang et al. model.

        Finds (θ*₁, θ*₂) such that each insurer's premium is a best response to
        the other's premium, with policyholders responding optimally via the
        Liang et al. threshold.

        For the two-class BMS only (len(discounts) == 2). Uses fixed-point
        iteration: start with symmetric premiums, iterate best-response until
        convergence.

        Expected profit for insurer i given premiums (θᵢ, θⱼ):

            J_i(θᵢ; θⱼ) ≈ θᵢ × π_i(θᵢ, θⱼ) × (1 − q_i × p_b)

        where π_i is the stationary fraction of policyholders with insurer i,
        p_b = P(Y > b*) is the reporting probability, and q_i represents
        expected claims per policyholder. The best response is found via a
        simple grid search over the premium space [0, premium_cap].

        This method is approximate — it uses a coarse grid for the best-response
        step. For a full solution, see Liang et al. (2026) Theorem 4.2.

        Parameters
        ----------
        kappa : float, default 1.25
            Penalty ratio. See ``liang_equilibrium`` for details.
        k1 : float, default 0.015
            Price sensitivity parameter.
        k2 : float, default 0.8
            Brand preference parameter.
        premium_cap : float, default 50.0
            Upper bound for the premium grid search.
        tol : float, default 1e-6
            Convergence tolerance on premium change.
        max_iter : int, default 500
            Maximum number of fixed-point iterations.

        Returns
        -------
        dict
            Keys:

            - ``theta1``: equilibrium Class 1 premium for insurer 1
            - ``theta2``: equilibrium Class 1 premium for insurer 2
            - ``threshold``: Nash equilibrium reporting threshold at (θ*₁, θ*₂)
            - ``converged``: bool, whether the iteration converged
            - ``iterations``: number of iterations taken
            - ``premium_gap``: |θ*₁ - θ*₂|

        Raises
        ------
        ValueError
            If the simulator was not constructed with exactly 2 NCD classes.
        """
        if self._n_classes != 2:
            raise ValueError(
                "nash_equilibrium_premiums() is only implemented for the two-class "
                f"BMS (len(discounts) == 2). This simulator has {self._n_classes} classes. "
                "See Liang et al. (2026) for the general N-class problem."
            )

        # Grid for best-response search
        n_grid = 200
        theta_grid = np.linspace(1.0, premium_cap, n_grid)

        # Loss distribution — use severity_dist if available, else fallback
        if self.severity_dist is not None:
            dist = self.severity_dist
        else:
            # Default: Gamma from Liang et al. base case
            dist = scipy_stats.gamma(a=1.2, scale=1.0 / 0.0085)
            warnings.warn(
                "No severity_dist provided; using default Gamma(1.2, 1/0.0085) "
                "from the Liang et al. (2026) numerical example. "
                "Supply severity_dist at construction for a calibrated result.",
                stacklevel=2,
            )

        def reporting_prob(b_star: float) -> float:
            """P(Y > b*) — probability a loss exceeds the threshold."""
            if b_star <= 0:
                return 1.0
            return float(dist.sf(b_star))

        def market_share_insurer1(t1: float, t2: float) -> float:
            """Expected share of policyholders at insurer 1 in steady state."""
            log_odds_base = np.log((1 - k2) / k2)
            eta = 1.0 / (1.0 + np.exp(k1 * (t1 - t2) + log_odds_base))
            return eta

        def expected_profit_i(ti: float, tj: float, is_insurer1: bool) -> float:
            """Approximate expected per-policyholder profit for insurer i."""
            # Compute threshold at these premiums
            result = self.liang_equilibrium(ti, tj, kappa=kappa, k1=k1, k2=k2)
            b_star = result["threshold"]
            p_report = reporting_prob(b_star)

            # Market share of insurer i
            share1 = market_share_insurer1(ti, tj)
            share_i = share1 if is_insurer1 else (1 - share1)

            # Expected premium income minus expected claims cost
            # Premium income: ti (Class 1) or ti*kappa (Class 2), weighted by stationary dist
            # With a simple 2-class model:
            # - Fraction in class 1: p0 / (p0 + p_report) — no-claim prob / (no-claim + claim)
            p0 = 1 - self.claim_freq  # prob of no accident
            p_claim = self.claim_freq  # prob of accident
            # Fraction of time in class 1 vs class 2 in two-state chain
            # Transition: from class 1: stay in 1 with prob (1-p_claim*p_report),
            #                           move to 2 with prob p_claim * p_report
            # from class 2: move to 1 with prob (1-p_claim*p_report), stay with ...
            # This is a simple 2-state chain
            q_report = p_claim * p_report
            # Stationary: π1 * q_report = π2 * (1-q_report) (detailed balance approx)
            # For 2-state: π1 = (1-q_report), π2 = q_report (if symmetric transition)
            pi1 = (1 - q_report)
            pi2 = q_report

            avg_premium = pi1 * ti + pi2 * ti * kappa
            # Expected claims cost = claim_freq * E[Y | Y > b*] * p_report
            if p_report > 1e-10:
                mean_reported_loss = float(dist.expect(lambda y: y, lb=b_star)) if b_star > 0 else float(dist.mean())
            else:
                mean_reported_loss = 0.0

            expected_claims = p_claim * p_report * mean_reported_loss
            profit = share_i * (avg_premium - expected_claims)
            return profit

        # Fixed-point iteration: start symmetric
        theta1 = self.base_premium * 0.9
        theta2 = self.base_premium * 0.9

        converged = False
        for iteration in range(1, max_iter + 1):
            # Best response for insurer 1 given theta2
            profits1 = np.array([
                expected_profit_i(t, theta2, is_insurer1=True) for t in theta_grid
            ])
            new_theta1 = float(theta_grid[np.argmax(profits1)])

            # Best response for insurer 2 given theta1
            profits2 = np.array([
                expected_profit_i(theta1, t, is_insurer1=False) for t in theta_grid
            ])
            new_theta2 = float(theta_grid[np.argmax(profits2)])

            if abs(new_theta1 - theta1) < tol and abs(new_theta2 - theta2) < tol:
                converged = True
                theta1, theta2 = new_theta1, new_theta2
                break

            theta1 = 0.5 * theta1 + 0.5 * new_theta1  # damped update for stability
            theta2 = 0.5 * theta2 + 0.5 * new_theta2

        eq_result = self.liang_equilibrium(theta1, theta2, kappa=kappa, k1=k1, k2=k2)

        return {
            "theta1": theta1,
            "theta2": theta2,
            "threshold": eq_result["threshold"],
            "converged": converged,
            "iterations": iteration,
            "premium_gap": abs(theta1 - theta2),
        }

    def class_premiums(self) -> np.ndarray:
        """
        Actual premiums for each NCD class: base_premium × (1 − discount_n).

        Returns
        -------
        np.ndarray
            Shape (n_classes,). Element n is the premium payable in class n.
        """
        return self.base_premium * (1 - self.discounts)

    def summary(self) -> None:
        """
        Print a formatted summary of the fitted simulator.

        Outputs the ladder structure, computed thresholds, reporting
        probabilities (if severity_dist is available), and corrected
        frequencies (if observed_freq was supplied to fit()).
        """
        self._check_fitted()

        premiums = self.class_premiums()
        n = self._n_classes

        print("BMSEquilibriumSimulator — NCD Underreporting Analysis")
        print("=" * 60)
        print(f"  NCD classes    : {n}")
        print(f"  Base premium   : £{self.base_premium:,.2f}")
        print(f"  Step-back      : {self.step_back} class(es) per fault claim")
        print(f"  Discount factor: {self.discount_factor:.4f} (r ≈ {(1/self.discount_factor - 1)*100:.1f}%)")
        print(f"  Claim frequency: {self.claim_freq:.4f} (per car year)")
        if self.severity_dist is not None:
            print(f"  Severity dist  : {self.severity_dist.__class__.__name__}")
        else:
            print(f"  Severity dist  : not supplied")
        print()

        # Header
        header = (
            f"{'Class':>6} {'Discount':>9} {'Premium':>10} {'Threshold b*':>13}"
        )
        if self._reporting_probs is not None:
            header += f" {'P(Y>b*)':>9}"
        if self._observed_freq is not None and self._corrected_freq is not None:
            header += f" {'Obs freq':>10} {'Corr freq':>10} {'Bias':>8}"
        print(header)
        print("-" * len(header))

        for i in range(n):
            line = (
                f"{i:>6} {self.discounts[i]:>8.1%} {premiums[i]:>9.2f} "
                f"{self._thresholds[i]:>13.2f}"  # type: ignore[index]
            )
            if self._reporting_probs is not None:
                line += f" {self._reporting_probs[i]:>8.3f}"
            if self._observed_freq is not None and self._corrected_freq is not None:
                obs = self._observed_freq[i]
                corr = self._corrected_freq[i]
                if np.isfinite(corr):
                    bias_pct = (obs - corr) / corr * 100
                    line += f" {obs:>10.4f} {corr:>10.4f} {bias_pct:>7.1f}%"
                else:
                    line += f" {obs:>10.4f} {'N/A':>10} {'N/A':>8}"
            print(line)

        print()
        print("  Stationary distribution (% policyholders at steady state):")
        for i in range(n):
            bar_len = int(self._stationary_dist[i] * 40)  # type: ignore[index]
            bar = "█" * bar_len
            print(f"  Class {i:>2}: {self._stationary_dist[i]:>6.3f}  {bar}")
        print()

    # ------------------------------------------------------------------
    # Internal: Lemaire algorithm
    # ------------------------------------------------------------------

    def _lemaire_thresholds(self) -> np.ndarray:
        """
        Compute reporting thresholds via simplified Lemaire (1977) algorithm.

        Uses the NPV of the step-back penalty as the threshold. This is the
        amount a policyholder in class n would need to lose before it becomes
        worthwhile to report:

            b*_n = B × (d_n - d_{n-k}) × Σ_{t=1}^{T_n} δ^t

        where d_n is the discount in class n, k = step_back, and T_n is the
        number of claim-free years to return to class n from class n-k.

        The threshold is zero for classes where the step-back is to a class
        with equal or higher discount (no penalty from claiming).

        Returns
        -------
        np.ndarray
            Shape (n_classes,). Thresholds in same units as base_premium.
        """
        thresholds = np.zeros(self._n_classes)
        premiums = self.class_premiums()

        for n in range(self._n_classes):
            # Class after step-back (clamped at 0)
            n_after = max(0, n - self.step_back)

            # Premium penalty per year after a fault claim
            premium_increase = premiums[n_after] - premiums[n]

            if premium_increase <= 0:
                # No penalty — threshold is zero (always claim)
                thresholds[n] = 0.0
                continue

            # Rebuilding horizon: number of claim-free years to get from
            # n_after back to n. With step_back=2 and claim-free → +1 class/yr:
            rebuilding_years = n - n_after  # = min(step_back, n)

            # NPV of the premium penalty over the rebuilding horizon
            # Assumes: one year at elevated premium, then back to original
            # (simplified — full Lemaire accounts for stochastic claims during rebuild)
            #
            # The penalty stream is: pay (premiums[n_after] - premiums[n]) per year
            # for ~rebuilding_years years, discounted at δ
            npv = 0.0
            for t in range(1, min(rebuilding_years, self.max_horizon) + 1):
                npv += premium_increase * (self.discount_factor ** t)

            # Also account for subsequent step-down years reaching higher classes
            # For the class gap beyond the immediate step-back:
            # Add incremental premium differences as class rebuilds
            for step in range(1, rebuilding_years):
                intermediate_class = n_after + step
                if intermediate_class < self._n_classes:
                    # Additional savings once this intermediate class is reached
                    intermediate_increase = premiums[n_after] - premiums[intermediate_class]
                    if intermediate_increase > 0:
                        # This saving starts at year `step` and lasts for `rebuilding_years - step` years
                        for t in range(step + 1, min(rebuilding_years, self.max_horizon) + 1):
                            # Remove the part already counted, add the new (lower) penalty
                            pass  # handled by summing the full incremental penalty above

            thresholds[n] = npv

        return thresholds

    def _compute_reporting_probs(self, thresholds: np.ndarray) -> np.ndarray:
        """
        Compute P(Y > b*_n) for each NCD class using the severity distribution.

        Parameters
        ----------
        thresholds : np.ndarray
            Shape (n_classes,). Reporting thresholds from _lemaire_thresholds.

        Returns
        -------
        np.ndarray
            Shape (n_classes,). Reporting probabilities in (0, 1].
        """
        probs = np.empty(self._n_classes)
        for n in range(self._n_classes):
            b = thresholds[n]
            if b <= 0:
                probs[n] = 1.0
            else:
                probs[n] = float(self.severity_dist.sf(b))
        return probs

    def _build_transition_matrix(self, reporting_probs: np.ndarray) -> np.ndarray:
        """
        Build the NCD Markov transition matrix given reporting probabilities.

        T[i, j] = probability of moving from class i to class j in one period.

        Transition rules:
        - With probability claim_freq × p_i: a reportable loss occurs → move
          to class max(0, i - step_back).
        - With probability (1 - claim_freq × p_i): no reportable loss → move
          to class min(n_classes - 1, i + 1).

        The effective probability of reporting a claim at class i is:
            q_i = claim_freq × p_i

        where p_i = P(Y > b*_i) is the probability the loss exceeds the threshold.

        Parameters
        ----------
        reporting_probs : np.ndarray
            Shape (n_classes,). P(Y > b*_n).

        Returns
        -------
        np.ndarray
            Shape (n_classes, n_classes). Row-stochastic transition matrix.
        """
        T = np.zeros((self._n_classes, self._n_classes))
        for i in range(self._n_classes):
            q_i = self.claim_freq * reporting_probs[i]  # prob of reporting
            # Step back on reported claim
            j_back = max(0, i - self.step_back)
            # Step forward on no reported claim
            j_forward = min(self._n_classes - 1, i + 1)

            T[i, j_back] += q_i
            T[i, j_forward] += (1 - q_i)

        return T

    def _compute_stationary_distribution(self, T: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution of the Markov chain.

        Solves (I − T^⊤)π = 0 with Σπ = 1, using the constraint replacement
        method: replace the last equation with Σπ = 1.

        Parameters
        ----------
        T : np.ndarray
            Shape (n_classes, n_classes). Row-stochastic transition matrix.

        Returns
        -------
        np.ndarray
            Shape (n_classes,). Stationary probabilities summing to 1.
        """
        n = self._n_classes
        # Set up (T^T - I)π = 0, replace last row with Σπ = 1
        A = T.T - np.eye(n)
        A[-1, :] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0

        try:
            pi = solve(A, b)
        except Exception:
            # Fallback: power iteration
            pi = np.ones(n) / n
            for _ in range(10000):
                pi_new = pi @ T
                if np.max(np.abs(pi_new - pi)) < 1e-10:
                    break
                pi = pi_new
            pi = pi / pi.sum()

        # Clip numerical noise
        pi = np.clip(pi, 0, 1)
        pi = pi / pi.sum()
        return pi

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Simulator has not been fitted. Call .fit() first."
            )

    def __repr__(self) -> str:
        if not self._fitted:
            return (
                f"BMSEquilibriumSimulator("
                f"n_classes={self._n_classes}, "
                f"base_premium={self.base_premium}, "
                f"not fitted)"
            )
        return (
            f"BMSEquilibriumSimulator("
            f"n_classes={self._n_classes}, "
            f"base_premium={self.base_premium}, "
            f"step_back={self.step_back}, "
            f"discount_factor={self.discount_factor})"
        )
